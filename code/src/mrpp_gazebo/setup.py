from glob import glob

from setuptools import find_packages, setup

package_name = "mrpp_gazebo"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/worlds", glob("worlds/*.sdf")),
        (f"share/{package_name}/models/pedestrian", glob("models/pedestrian/*.sdf")),
        (
            f"share/{package_name}/models/pedestrian_animated",
            glob("models/pedestrian_animated/*.sdf"),
        ),
        (
            f"share/{package_name}/models/pedestrian_animated/meshes",
            glob("models/pedestrian_animated/meshes/*.dae"),
        ),
        (
            f"share/{package_name}/models/walking_person",
            glob("models/walking_person/*.sdf"),
        ),
        (
            f"share/{package_name}/models/walking_person/meshes",
            glob("models/walking_person/meshes/*.dae"),
        ),
        (
            f"share/{package_name}/models/person_walking",
            glob("models/person_walking/*.sdf"),
        ),
        (
            f"share/{package_name}/models/person_walking/meshes",
            glob("models/person_walking/meshes/*.dae"),
        ),
        (f"share/{package_name}/models/person_actor", glob("models/person_actor/*.sdf")),
        (
            f"share/{package_name}/models/person_actor/meshes",
            glob("models/person_actor/meshes/*.dae"),
        ),
        (
            f"share/{package_name}/models/mecanum_robot",
            glob("models/mecanum_robot/*"),
        ),
        (
            f"share/{package_name}/models/turtlebot3_burger",
            glob("models/turtlebot3_burger/*"),
        ),
        (
            f"share/{package_name}/models/turtlebot3_common/meshes/bases",
            glob("models/turtlebot3_common/meshes/bases/*"),
        ),
        (
            f"share/{package_name}/models/turtlebot3_common/meshes/wheels",
            glob("models/turtlebot3_common/meshes/wheels/*"),
        ),
        (
            f"share/{package_name}/models/turtlebot3_common/meshes/sensors",
            glob("models/turtlebot3_common/meshes/sensors/*"),
        ),
        (
            f"share/{package_name}/models/pedestrian_fuel",
            glob("models/pedestrian_fuel/*.sdf"),
        ),
        (
            f"share/{package_name}/models/pedestrian_fuel/meshes",
            glob("models/pedestrian_fuel/meshes/*.dae"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="lixiang-886",
    maintainer_email="884237172@qq.com",
    description="Gazebo integration for multi-mobile-robot path planning experiments.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "spawn_robots = mrpp_gazebo.spawn_robots:main",
            "pedestrian_controller = mrpp_gazebo.pedestrian_controller:main",
        ]
    },
)
