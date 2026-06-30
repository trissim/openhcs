from openhcs.core.alias_property import AliasProperty


class AliasExample:
    value = "source"
    alias = AliasProperty[str]("value")


def test_alias_property_projects_source_attribute() -> None:
    instance = AliasExample()

    assert instance.alias == "source"
    assert AliasExample.alias.source_name == "value"
