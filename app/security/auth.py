import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import Header, HTTPException

load_dotenv()

_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:@-]{1,128}$")
_ALLOWED_VISIBILITY = {"private", "department", "tenant", "internal", "public"}


@dataclass
class AuthContext:
    user_id: str
    tenant_id: str
    department_id: str
    visibility: str = "tenant"
    is_mock: bool = False
    has_authorization: bool = False
    auth_mode: str = "trusted_headers"

    def normalized_visibility(self) -> str:
        if self.visibility == "internal":
            return "tenant"
        return self.visibility

    def to_state_fields(self) -> Dict[str, str]:
        return {
            "user_id": self.user_id,
            "created_by": self.user_id,
            "tenant_id": self.tenant_id,
            "department_id": self.department_id,
            "visibility": self.normalized_visibility(),
        }

    def to_log_dict(self) -> Dict[str, str]:
        payload = asdict(self)
        payload.pop("has_authorization", None)
        return payload


def _is_true(value: str, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _validate_identity_field(field_name: str, value: str) -> str:
    if not value:
        raise HTTPException(status_code=401, detail=f"缺少鉴权字段: {field_name}")
    if not _SAFE_ID_PATTERN.match(value):
        raise HTTPException(status_code=400, detail=f"鉴权字段格式非法: {field_name}")
    return value


def _normalize_visibility(raw_visibility: str) -> str:
    if not raw_visibility:
        return "tenant"
    visibility = raw_visibility.strip().lower()
    if visibility not in _ALLOWED_VISIBILITY:
        raise HTTPException(status_code=400, detail="visibility 非法")
    if visibility == "internal":
        return "tenant"
    return visibility


def redact_authorization_header(authorization: str) -> str:
    if not authorization:
        return ""
    if len(authorization) <= 16:
        return "***"
    return authorization[:8] + "***" + authorization[-4:]


def build_milvus_security_expr(auth: AuthContext) -> str:
    tenant_id = auth.tenant_id
    department_id = auth.department_id
    user_id = auth.user_id
    # 优化: 优化的原因是仅做 tenant_id 等值过滤过于脆弱，无法解释 public / department / private 的访问边界，因此补成分层可见性表达式。
    expr = (
        f'(visibility == "public") or '
        f'(tenant_id == "{tenant_id}" and visibility == "tenant") or '
        f'(tenant_id == "{tenant_id}" and department_id in ["{department_id}", "default"] and visibility == "department") or '
        f'(tenant_id == "{tenant_id}" and created_by == "{user_id}" and visibility == "private")'
    )
    return f"({expr})"


def build_mongo_security_query(auth: AuthContext) -> Dict[str, List[Dict[str, str]]]:
    return {
        "$or": [
            {"visibility": "public"},
            {"tenant_id": auth.tenant_id, "visibility": {"$in": ["tenant", "internal"]}},
            {"tenant_id": auth.tenant_id, "department_id": {"$in": [auth.department_id, "default"]}, "visibility": "department"},
            {"tenant_id": auth.tenant_id, "created_by": auth.user_id, "visibility": "private"},
        ]
    }


async def get_auth_context(
    authorization: str = Header(default=""),
    x_user_id: str = Header(default="", alias="X-User-Id"),
    x_tenant_id: str = Header(default="", alias="X-Tenant-Id"),
    x_department_id: str = Header(default="", alias="X-Department-Id"),
) -> AuthContext:
    allow_anonymous = _is_true(os.getenv("AUTH_ALLOW_ANONYMOUS", "true"))
    default_visibility = _normalize_visibility(os.getenv("AUTH_DEFAULT_VISIBILITY", "tenant"))
    auth_mode = os.getenv("AUTH_MODE", "trusted_headers")
    has_authorization = bool(authorization)

    if x_user_id or x_tenant_id or x_department_id:
        user_id = _validate_identity_field("X-User-Id", x_user_id)
        tenant_id = _validate_identity_field("X-Tenant-Id", x_tenant_id)
        department_id = _validate_identity_field("X-Department-Id", x_department_id or "default")
        return AuthContext(
            user_id=user_id,
            tenant_id=tenant_id,
            department_id=department_id,
            visibility=default_visibility,
            is_mock=False,
            has_authorization=has_authorization,
            auth_mode=auth_mode,
        )

    if allow_anonymous:
        return AuthContext(
            user_id=os.getenv("AUTH_MOCK_USER_ID", "dev-user"),
            tenant_id=os.getenv("AUTH_MOCK_TENANT_ID", "default"),
            department_id=os.getenv("AUTH_MOCK_DEPARTMENT_ID", "default"),
            visibility=default_visibility,
            is_mock=True,
            has_authorization=has_authorization,
            auth_mode="mock",
        )

    raise HTTPException(status_code=401, detail="缺少鉴权信息，当前环境不允许匿名访问")
