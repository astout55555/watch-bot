import pytest

from watchbot.billref import (
    BillRef,
    canonical_bill_id,
    display_name,
    parse_bill_id,
    question_reference_variants,
)


class TestCanonicalBillId:
    def test_simple_house_bill(self):
        assert canonical_bill_id("HR", 1181, 119) == "hr1181-119"

    def test_simple_senate_bill(self):
        assert canonical_bill_id("S", 442, 119) == "s442-119"

    def test_congress_gov_style_types_map_directly(self):
        assert canonical_bill_id("HJRES", 27, 119) == "hjres27-119"
        assert canonical_bill_id("SCONRES", 5, 119) == "sconres5-119"

    def test_dotted_types_are_normalized(self):
        assert canonical_bill_id("H.R.", 1181, 119) == "hr1181-119"
        assert canonical_bill_id("S.J.Res.", 12, 119) == "sjres12-119"
        assert canonical_bill_id("H. Con. Res.", 5, 119) == "hconres5-119"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            canonical_bill_id("XYZ", 1, 119)

    def test_nonpositive_number_raises(self):
        with pytest.raises(ValueError):
            canonical_bill_id("HR", 0, 119)


class TestParseBillId:
    def test_round_trip(self):
        ref = parse_bill_id("hr1181-119")
        assert ref == BillRef(bill_type="hr", number=1181, congress=119)
        assert canonical_bill_id(ref.bill_type, ref.number, ref.congress) == "hr1181-119"

    def test_multiword_type(self):
        assert parse_bill_id("hconres5-119") == BillRef("hconres", 5, 119)

    def test_malformed_raises(self):
        for bad in ["hr-119", "1181-119", "hr1181", "xyz1-119", ""]:
            with pytest.raises(ValueError):
                parse_bill_id(bad)


class TestQuestionReferenceVariants:
    """Vote `question` text in GovQL embeds bill references in two styles:
    House clerk style ("H R 1181") and Senate/dotted style ("H.R. 1181")."""

    def test_house_bill_variants(self):
        variants = question_reference_variants(BillRef("hr", 1181, 119))
        assert "H R 1181" in variants
        assert "H.R. 1181" in variants

    def test_senate_bill_variants(self):
        variants = question_reference_variants(BillRef("s", 442, 119))
        assert "S 442" in variants
        assert "S. 442" in variants

    def test_joint_resolution_variants(self):
        variants = question_reference_variants(BillRef("hjres", 27, 119))
        assert "H J RES 27" in variants
        assert "H.J.Res. 27" in variants


class TestDisplayName:
    def test_display_forms(self):
        assert display_name(BillRef("hr", 1181, 119)) == "H.R. 1181"
        assert display_name(BillRef("s", 442, 119)) == "S. 442"
        assert display_name(BillRef("sjres", 12, 119)) == "S.J.Res. 12"
        assert display_name(BillRef("hconres", 5, 119)) == "H.Con.Res. 5"
