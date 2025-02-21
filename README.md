# A Quantization Framework for PyTorch
A framework for fake quantization in PyTorch implementing several quantization-aware and post-training quantization method. Additionally, it implements the following quantization methods:

+ Kirtas, Manos, Athina Oikonomou, Nikolaos Passalis, George Mourgias-Alexandris, Miltiadis Moralis-Pegios, Nikos Pleros, and Anastasios Tefas. "[Quantization-aware training for low precision photonic neural networks.](https://www.sciencedirect.com/science/article/abs/pii/S0893608022003598)" Neural Networks 155 (2022): 561-573. 
+ Kirtas, Manos, Nikolaos Passalis, Athina Oikonomou, Miltos Moralis-Pegios, George Giamougiannis, Apostolos Tsakyridis, George Mourgias-Alexandris, Nikolaos Pleros, and Anastasios Tefas. "[Mixed-precision quantization-aware training for photonic neural networks.](https://link.springer.com/article/10.1007/s00521-023-08848-8)" Neural Computing and Applications 35, no. 29 (2023): 21361-21379. 

| Methods         | Post Training | Quantization Aware | Related Code                                                             |
|-----------------|---------------|--------------------|--------------------------------------------------------------------------|
| Normalized      | ✓             | ✓                  | [normalized.py](./torch_fquant/v2/observers/normalized.py)               |
| Moving Average  | ✓             | ✓                  | [movingaverage.py](./torch_fquant/v2/observers/movingaverage.py)         |
| MinMax          | ✓             | ✓                  | [minmax.py](./torch_fquant/v2/observers/minmax.py)                       |
| MinMaxSTD       | ✓             | ✓                  | [minmaxstd.py](./torch_fquant/v2/observers/minmax.py)                    |
| SimplerMinMax   | ✓             | ✓                  | [minmax_simpler.py](./torch_fquant/v2/observers/minmax_simpler.py)       |
| Mixed Precision |               | ✓                  | [gaussian_qscheduler.py](./torch_fquant/v2/mixed/gaussian_qscheduler.py) |


Demonstrations for the methodologies can be found in:
+ [eakirtas/quantization_demo](https://github.com/eakirtas/quantization_demo)
+ [eakirtas/gradual_mixed_quant](https://github.com/eakirtas/gradual_mixed_quant)

## Citations

```
@article{kirtas2022quantization,
  title={Quantization-aware training for low precision photonic neural networks},
  author={Kirtas, Manos and Oikonomou, Athina and Passalis, Nikolaos and Mourgias-Alexandris, George and Moralis-Pegios, Miltiadis and Pleros, Nikos and Tefas, Anastasios},
  journal={Neural Networks},
  volume={155},
  pages={561--573},
  year={2022},
  publisher={Elsevier}
}
```

```
@article{kirtas2023mixed,
  title={Mixed-precision quantization-aware training for photonic neural networks},
  author={Kirtas, Manos and Passalis, Nikolaos and Oikonomou, Athina and Moralis-Pegios, Miltos and Giamougiannis, George and Tsakyridis, Apostolos and Mourgias-Alexandris, George and Pleros, Nikolaos and Tefas, Anastasios},
  journal={Neural Computing and Applications},
  volume={35},
  number={29},
  pages={21361--21379},
  year={2023},
  publisher={Springer}
}
```



## Acknowledgements

This project has received funding from the European Union’s Horizon 2020 research and innovation program under Grant Agreement No 871391 (PlasmoniAC). This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains. 
