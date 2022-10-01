from pytest import fixture


@fixture(scope="module")
def data_path() -> str:
    return "data_trash"
