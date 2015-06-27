# Automatic Instrument Identification in Polyphonic Music

Course project for DS-GA1003 Machine Learning.

### Members:

- Jiyuan Qian
- Tian Wang
- Peter Li

### Dataset:

MedleyDB, which is available from http://medleydb.weebly.com/

### Results

We compared Convnets trained on raw audio, MFCC and CQT with
traditional MIR methods that extracts Gaussian features from MFCC and
its first and second order deltas. Convnets trained on handcrafted
features can outperform traditional methods, and that trained on raw
audio, though takes much longer training time, can achieve arguably
better performance.
