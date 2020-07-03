# -*- coding: utf-8 -*-
"""
Created by etayupanta at 7/1/2020 - 14:44
PyCharm - DeepVO
__author__ = 'Eduardo Tayupanta'
__email__ = 'etayupanta@yotec.tech'
"""

# Import Libraries:
import torch


def main():
    x = torch.randn(2, 3, 5)
    print(x)
    print(x.size())
    y = x.permute(2, 0, 1)
    print(y)
    print(y.size())


if __name__ == "__main__":
    main()
