from testfixtures import compare
from unittest import TestCase

import entity


class TestIntelligenceID(TestCase):

    def test_create(self):
        id = entity.IntelligenceID.create()
        self.assertRegexpMatches(id, r"^[0-9a-z]{32}_[\d]{18}$")
