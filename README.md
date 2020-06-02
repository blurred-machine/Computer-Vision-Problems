# Computer-Vision-Problems
In this repository I have solved two problems of computer vision. First Implementation is building a CNN in which the input is a noisy number and the output is a denoised number based on AutoEncoders. Second implementations is sequencing the characters read from number plates done using obect detection.

* Please use the jupyter notebook to run the above files in iPython environment.
* or directly run the `Problem_x.py` file in the python shell.

## Implementation P1
- Loss Calculation during training:
- ![Loss Calculation](https://github.com/blurred-machine/Computer-Vision-Problems/blob/master/screenshots/training%20loss.PNG)

- Loss Calculation during training:
- ![Accuracy calculation](https://github.com/blurred-machine/Computer-Vision-Problems/blob/master/screenshots/training%20accuracy.PNG)

- Final Reult with accuracy of **81.27%** on Training and **81.35%** on Validation Set:
- ![Result](https://github.com/blurred-machine/Computer-Vision-Problems/blob/master/screenshots/final%20result.PNG)


## Implementation P2
Steps followed:
- Extracted the characters from the XML result of object detection using `xml.etree.ElementTree`.
- Wrote functions to extract **character** and **boundung boxes** data from XML root.
- Built a sorting sequence algorithm for characters based on bounding box coordinates.
- Tested and verified the algorithm on single line and double line number plates.

Thank You!
