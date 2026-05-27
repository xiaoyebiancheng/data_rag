import asyncio
import os
import unittest

from app.security.auth import build_milvus_security_expr, build_mongo_security_query, get_auth_context


class TestAuthContext(unittest.TestCase):
    def test_header_auth(self):
        os.environ["AUTH_ALLOW_ANONYMOUS"] = "false"
        context = asyncio.run(
            get_auth_context(
                authorization="Bearer abc",
                x_user_id="user_1",
                x_tenant_id="tenant_a",
                x_department_id="dept_x",
            )
        )
        self.assertEqual(context.user_id, "user_1")
        self.assertEqual(context.tenant_id, "tenant_a")
        self.assertFalse(context.is_mock)

    def test_mock_auth(self):
        os.environ["AUTH_ALLOW_ANONYMOUS"] = "true"
        os.environ["AUTH_MOCK_USER_ID"] = "dev-user"
        os.environ["AUTH_MOCK_TENANT_ID"] = "default"
        os.environ["AUTH_MOCK_DEPARTMENT_ID"] = "default"
        context = asyncio.run(
            get_auth_context(
                authorization="",
                x_user_id="",
                x_tenant_id="",
                x_department_id="",
            )
        )
        self.assertEqual(context.user_id, "dev-user")
        self.assertEqual(context.tenant_id, "default")
        self.assertTrue(context.is_mock)

    def test_security_filters(self):
        os.environ["AUTH_ALLOW_ANONYMOUS"] = "false"
        context = asyncio.run(
            get_auth_context(
                authorization="Bearer abc",
                x_user_id="user_1",
                x_tenant_id="tenant_a",
                x_department_id="dept_x",
            )
        )
        milvus_expr = build_milvus_security_expr(context)
        mongo_query = build_mongo_security_query(context)
        self.assertIn('tenant_id == "tenant_a"', milvus_expr)
        self.assertIn("$or", mongo_query)
        self.assertEqual(len(mongo_query["$or"]), 4)


if __name__ == "__main__":
    unittest.main()
