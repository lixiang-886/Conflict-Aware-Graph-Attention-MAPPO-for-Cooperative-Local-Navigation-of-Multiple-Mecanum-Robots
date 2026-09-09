from setuptools import find_packages, setup

package_name = "mrpp_rl"

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
    description="Reinforcement learning algorithms for cooperative multi-robot path planning.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "train_mo_gat_mappo = mrpp_rl.train:main",
            "evaluate_mrpp = mrpp_rl.evaluate:main",
        ]
    },
)
