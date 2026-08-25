# 开发文档

## 安装

```bash
# 注意要用虚拟环境，不要覆盖主机的原版 deep_gemm
source env3.12/bin/activate

# JIT需要用的头文件
ln -sfn "$(pwd)/third-party/cutlass/include/cutlass" deep_gemm/include/
ln -sfn "$(pwd)/third-party/cutlass/include/cute" deep_gemm/include/

rm -rf build dist *.egg-info
python setup.py bdist_wheel
python -m pip install --force-reinstall dist/deep_gemm_cpp-*.whl
```

## 单测

```bash
source env3.12/bin/activate
python tests_overlap/test_gemm_baseline.py
```


## 开发进展

8.25: TODO

