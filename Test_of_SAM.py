#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:34:18 2024

@author: gustav-emilbechlarsen
"""

import cv2


test_image = cv2.imread("/Users/gustav-emilbechlarsen/Desktop/test1.jpeg")


from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["vit_h"](checkpoint="/Users/gustav-emilbechlarsen/Downloads/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(test_image)



