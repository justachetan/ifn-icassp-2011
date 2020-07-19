# Iterative feature normalization

This repository contains a Python implementation for the paper:

>[Iterative Feature Normalisation for Emotional Speech Recognition *Carlos Busso*, *Angeliki Metallinou* and *Shrikanth S. Narayanan*  *ICASSP 2011*](https://ieeexplore.ieee.org/document/5947652)

The code can be used to generate the results given in the paper on the [RAVDESS](https://zenodo.org/record/1188976#.XnrzbYAzbkw) dataset.

## Running the code

The detailed steps for running the script have been described below:

### Data Pre-processing

- First download the [RAVDESS dataset](https://zenodo.org/record/1188976#.XnrzbYAzbkw) and unzip it.
- Now, you can use `dataprocessor.py` to process this dataset into the input format required by the IFN script.

```
$ python3 dataprocessor.py --help

usage: dataprocessor.py [-h] [--data_dir DATA_DIR] [--out_emo OUT_EMO]
                        [--out_ref OUT_REF]

Preprocessing script for the RAVDESS dataset for IFN input

optional arguments:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Main directory of the RAVDESS dataset
  --out_emo OUT_EMO    Path where emotional corpus will be saved
  --out_ref OUT_REF    Path where neutral reference corpus will be saved

```

- The reference corpus will consist of a single `neutral` class clip, randomly selected from each speaker (so 24 data points).
- The rest of the file go into the emotional corpus.

### Running IFN

- Now that we have the data files processed, it is time to run IFN!
- You can use `ifn.py` to run the normalization procedure on the dumped corpus.

```
$ python3 ifn.py --help
usage: ifn.py [-h] --ref_corpus REF_CORPUS --emo_corpus EMO_CORPUS
              [--analysis] --log_dir LOG_DIR [--enable_neutral] [--feat FEAT]
              [--max_iter MAX_ITER] [--ss_iter SS_ITER] [--nlabel NLABEL]
              [--elabel ELABEL] [--test_size TEST_SIZE]
              [--norm_scheme NORM_SCHEME] [--neu_threshold NEU_THRESHOLD]
              [--spkr_file_threshold SPKR_FILE_THRESHOLD]
              [--switch_threshold SWITCH_THRESHOLD]

Script for running Iterative Feature Normalization.

optional arguments:
  -h, --help            show this help message and exit
  --ref_corpus REF_CORPUS
                        dump of the reference corpus
  --emo_corpus EMO_CORPUS
                        dump of the emotional corpus
  --analysis            if you want to generate detailed results for your run,
                        similar to the paper.
  --log_dir LOG_DIR     log directory where you want to dump all results
  --enable_neutral      follow "neutral" scheme for training
  --feat FEAT           list of features to be used. See README for options.
  --max_iter MAX_ITER   maximum no. of iterations for an IFN simulation
  --ss_iter SS_ITER     number of IFN simulations to perform. (only matters
                        with --analysis flag)
  --nlabel NLABEL       label for neutral instance
  --elabel ELABEL       label for emotioal instance
  --test_size TEST_SIZE
                        test size at the time of analysis. (only matters with
                        --analysis flag)
  --norm_scheme NORM_SCHEME
                        normalization scheme ('ifn', 'none', 'opt', 'global')
  --neu_threshold NEU_THRESHOLD
                        likelihood threshold for claddifying an instance as
                        neutral
  --spkr_file_threshold SPKR_FILE_THRESHOLD
                        minimum number of files to be classified as neutral
                        for each speaker
  --switch_threshold SWITCH_THRESHOLD
                        minimum files to switch for IFN to continue
``` 

#### Important notes

- The `--norm_scheme` can have 4 values - `ifn`, `none`, `opt` and `global`. The details for each of these schemes are available in the paper.
- `--feat` flag has to be a comma-separated list of features. The features that can be used are described in `features.py`. To know more in detail about what they mean, refer to the paper. 
- You can add your own features by adding a function in `features.py` and making a corresponding entry in the `feature_map`.


## Results

For detailed results and analysis, please refer to the [report](./report.pdf)


- - -

This code was written as a part of a course assignment in **Affective Computing (CSE661)** with [Dr. Jainendra Shukla](https://www.iiitd.ac.in/jainendra) at IIIT Delhi during Winter 2020 Semester. 

For bugs in the code, please write to: aditya16217 [at] iiitd [dot] ac [dot] in
