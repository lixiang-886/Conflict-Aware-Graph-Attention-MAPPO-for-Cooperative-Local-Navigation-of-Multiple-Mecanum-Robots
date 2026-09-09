from glob import glob

from setuptools import find_packages, setup

package_name = "mrpp_experiments"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config/scenarios", glob("config/scenarios/*.yaml")),
        (f"share/{package_name}/config/methods", glob("config/methods/*.csv")),
        (f"share/{package_name}/config/methods", glob("config/methods/*.json")),
        (f"share/{package_name}/config/sensors", glob("config/sensors/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lixiang-886",
    maintainer_email="884237172@qq.com",
    description="Experiment configuration and metric tooling for multi-robot path planning.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "generate_dry_run_results = mrpp_experiments.dry_run_results:main",
            "make_experiment_plan = mrpp_experiments.experiment_matrix:main",
            "make_result_templates = mrpp_experiments.baseline_results:main",
            "merge_mrpp_results = mrpp_experiments.merge_results:main",
            "run_mrpp_batch = mrpp_experiments.run_batch:main",
        ]
    },
)
