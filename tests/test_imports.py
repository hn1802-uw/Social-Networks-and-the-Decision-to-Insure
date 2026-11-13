"""Smoke tests to verify all modules can be imported."""


def test_import_data():
    from auxiliary import data

    assert hasattr(data, "load_analysis_data")


def test_import_estimation():
    from auxiliary import estimation

    for name in (
        "regression_table",
        "eq1_network_model",
        "eq2_price_model",
        "eq3_knowledge_model",
        "eq4_info_iv_model",
    ):
        assert hasattr(estimation, name)


def test_import_plots():
    from auxiliary import plots

    assert hasattr(plots, "plot_dag")
