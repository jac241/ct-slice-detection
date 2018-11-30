
CT Slice Detection
==================


Code repository for the paper

Fahdi Kanavati, Shah Islam, Eric O. Aboagye, and Andrea Rockall: 
“Automatic L3 slice detection in 3D CT images using fully-convolutional networks”, 2018; arXiv:1811.09244.


To use, go into package directory and run

`python setup.py develop`

The training and testing scripts should become available from the console.

Update the config files in `configs/` directory so that the dataset parameter and output point to the right directory.

To run the trainer, execute 

`python l3_detect_trainer --config configs/slice_detect_1d.cfg`

To run the tester, execute 

`python l3_detect_tester --config configs/slice_detect_1d.cfg`



## Dataset

The dataset was collected from multiple publicly available datasets:

 1. 3 sets were obtained from [the Cancer Imaging Archive (TCIA)](http://www.cancerimagingarchive.net/): 
 
     - [head and neck](http://doi.org/10.7937/K9/TCIA.2017.umz8dv6s)
     - [ovarian](http://dx.doi.org/10.7937/K9/TCIA.2016.NDO1MDFQ) 
     - [colon](http://doi.org/10.7937/K9/TCIA.2015.NWTESAY1)
       
 2. a liver tumour dataset is obtained from the 
 [LiTS segmentation challenge](https://competitions.codalab.org/competitions/17094);
 
 3. an ovarian cancer dataset is obtained from Hammersmith Hospital (HH), London.

The dataset is available for download in MIPs format from 
[here](https://imperialcollegelondon.box.com/s/0vt07mxy0re4zwao0sk76ywdt2s1pclm)

The subset of transitional vertabrae cases can be 
[here](https://imperialcollegelondon.box.com/s/mw7ysamajjcp1ot0721e6nl36xku0acv)
