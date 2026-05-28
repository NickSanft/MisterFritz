"""Tests for the HMAC-signed chat cookie helpers."""
import unittest

import chat_auth


SECRET = "test-secret-32-bytes-long-enough-honest"


class TestMakeCookie(unittest.TestCase):
    def test_round_trip_returns_username(self):
        token = chat_auth.make_cookie("alice", SECRET)
        self.assertEqual(chat_auth.verify_cookie(token, SECRET), "alice")

    def test_empty_username_rejected(self):
        with self.assertRaises(ValueError):
            chat_auth.make_cookie("", SECRET)

    def test_empty_secret_rejected(self):
        with self.assertRaises(ValueError):
            chat_auth.make_cookie("alice", "")

    def test_payload_has_three_colon_separated_parts(self):
        token = chat_auth.make_cookie("alice", SECRET)
        self.assertEqual(token.count(":"), 2)


class TestVerifyCookie(unittest.TestCase):
    def test_none_token_returns_none(self):
        self.assertIsNone(chat_auth.verify_cookie(None, SECRET))

    def test_empty_token_returns_none(self):
        self.assertIsNone(chat_auth.verify_cookie("", SECRET))

    def test_wrong_secret_rejected(self):
        token = chat_auth.make_cookie("alice", SECRET)
        self.assertIsNone(chat_auth.verify_cookie(token, "different-secret"))

    def test_tampered_username_rejected(self):
        token = chat_auth.make_cookie("alice", SECRET)
        # Splice in a different username — signature won't match.
        _, ts, sig = token.split(":")
        tampered = f"eve:{ts}:{sig}"
        self.assertIsNone(chat_auth.verify_cookie(tampered, SECRET))

    def test_tampered_signature_rejected(self):
        token = chat_auth.make_cookie("alice", SECRET)
        user, ts, _ = token.split(":")
        tampered = f"{user}:{ts}:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        self.assertIsNone(chat_auth.verify_cookie(tampered, SECRET))

    def test_malformed_token_rejected(self):
        for token in ("alice", "alice:foo", "::", "a:b:c:d"):
            with self.subTest(token=token):
                self.assertIsNone(chat_auth.verify_cookie(token, SECRET))

    def test_non_numeric_timestamp_rejected(self):
        # Manually construct a token with a non-numeric ts.
        import hashlib
        import hmac
        sig = hmac.new(SECRET.encode(), b"alice:notanumber", hashlib.sha256).hexdigest()
        token = f"alice:notanumber:{sig}"
        self.assertIsNone(chat_auth.verify_cookie(token, SECRET))

    def test_expired_cookie_rejected(self):
        # Cookie minted 31 days ago.
        token = chat_auth.make_cookie("alice", SECRET, now=1_700_000_000)
        # "Now" is 31 days later — exceeds the 30-day max.
        self.assertIsNone(
            chat_auth.verify_cookie(token, SECRET, now=1_700_000_000 + 31 * 24 * 60 * 60)
        )

    def test_far_future_cookie_rejected(self):
        # Cookie minted 10 minutes in the "future" — likely tampering or clock skew.
        token = chat_auth.make_cookie("alice", SECRET, now=1_700_000_000 + 600)
        self.assertIsNone(chat_auth.verify_cookie(token, SECRET, now=1_700_000_000))


if __name__ == "__main__":
    unittest.main()
