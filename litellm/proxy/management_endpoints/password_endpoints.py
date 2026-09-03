"""
Self-service password management.

/user/password/change

Deliberately NOT wrapped in `management_endpoint_wrapper`: the wrapper emits
request kwargs to OTEL spans, which would log plaintext passwords.
"""

from typing import TYPE_CHECKING, Final

from fastapi import APIRouter, Depends, HTTPException

from litellm.proxy._types import (
    ChangePasswordRequest,
    ChangePasswordResponse,
    CommonProxyErrors,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.password_policy import (
    get_password_policy,
    raise_password_validation_error,
    validate_new_password,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import hash_password, verify_password
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.user_repository import UserRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient

router: Final = APIRouter()


def _user_table(
    prisma_client: "PrismaClient | None",
) -> "TableActions[prisma_models.LiteLLM_UserTable]":
    user_table: Final[TableActions[prisma_models.LiteLLM_UserTable]] = UserRepository(prisma_client).table
    return user_table


@router.post(
    "/user/password/change",
    tags=["Internal User management"],
    dependencies=(Depends(user_api_key_auth),),
)
async def change_password(
    data: ChangePasswordRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> ChangePasswordResponse:
    """
    Change the calling user's own password.

    Requires the current password. The new password must satisfy
    `general_settings.password_policy` (minimum length and, when enabled,
    breached-password screening via haveibeenpwned.com).

    Parameters:
    - current_password: str - The user's current password.
    - new_password: str - The password to change to.
    """
    from litellm.proxy.proxy_server import general_settings, prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    user_id: Final = user_api_key_dict.user_id
    if user_id is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "No user is associated with this session, so there is no password to change."},
        )

    user_row: Final = await _user_table(prisma_client).find_first(where={"user_id": user_id})
    stored_password: Final = user_row.password if user_row is not None else None
    if stored_password is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": (
                    "This account has no password set, so there is no password to change. "
                    "Passwords are set through an invitation link (POST /invitation/new)."
                )
            },
        )

    if not verify_password(data.current_password, stored_password):
        raise HTTPException(status_code=400, detail={"error": "Current password is incorrect."})

    password_failure: Final = await validate_new_password(
        password=data.new_password, policy=get_password_policy(general_settings)
    )
    if password_failure is not None:
        raise_password_validation_error(password_failure)

    await _user_table(prisma_client).update(
        where={"user_id": user_id},
        data={"password": hash_password(data.new_password)},
    )
    return ChangePasswordResponse(user_id=user_id, message="Password updated successfully.")
