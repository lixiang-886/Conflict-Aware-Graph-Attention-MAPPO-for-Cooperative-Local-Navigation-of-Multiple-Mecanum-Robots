from setuptools import find_packages, setup

package_name = "mrpp_paper"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lixiang-886",
    maintainer_email="884237172@qq.com",
    description="Paper figure and table generation tools for the multi-robot path planning study.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "plot_mrpp_metrics = mrpp_paper.plot_metrics:main",
            "plot_mrpp_trajectories = mrpp_paper.plot_trajectories:main",
            "mrpp_statistical_tests = mrpp_paper.statistical_tests:main",
        ]
    },
)
