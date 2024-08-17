from setuptools import setup, find_packages

setup(
    name='dense_sam',
    version='0.1.0',
    packages = find_packages(),  # 自动发现和包含所有包
    install_requires=[
        # 列出你的包依赖的其他包
    ],
    entry_points={
        'console_scripts': [
            # 如果你的包包含可执行命令行工具，可以在这里定义
        ],
    },
    author='Linyun Zhou',
    author_email='zhoulyaxx@zju.edu.cn',
    description='enhance sam with semantic',
    long_description='Long description of your package',
    url='https://github.com/your_username/your_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        # 添加适用的其他分类
    ],
)
