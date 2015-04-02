import pandas as pd
import copy
import librosa


def patch_label(start, end, time_windows, annotate, binary=False, threshold=None):
    """Labeling a patch given annotation
    Args:
        start(float): start time of a patch (in second)
        end(float): end time of a patch (in second)
        time_windows(float): time windows for average (in milliseconds)
        annotation(DataFrame): annotation dataframe for a specific song
        songname(string): song_name(string)
    Returns:
        label(pd.DataFrame):
            column: instrument
            index: label
            eg:
                        S01   S02
                label    1   0.93
    """
    #Transfer time to frame
    annotation = copy.copy(annotate)
    start_frame = librosa.time_to_frames(start, sr=1/0.0464, hop_length=1)+1
    end_frame = librosa.time_to_frames(end, sr=1/0.0464, hop_length=1)-1
    moving_frame = librosa.time_to_frames(time_windows/1000, sr=1/0.0464, hop_length=1)-librosa.time_to_frames(0, sr=1/0.0464, hop_length=1)
    #Pick annotation
    annotation = annotation.reset_index(drop=True)
    annotation.index += 1
    time_annot = annotation.loc[start_frame:end_frame].drop('time', 1)
    #Using maximum value of average in moving windows as label
    music_ins = time_annot.columns
    label = pd.DataFrame(index = ['label',], columns=music_ins)
    for j in range(len(time_annot.columns)):
        label_temp = max([sum(list(time_annot.ix[:, j])[i:i+moving_frame+1])/float(moving_frame+1)
                          for i in range(len(time_annot.ix[:, j])-moving_frame)])
        #binary output
        if binary and threshold:
            if label_temp >= threshold:
                label_temp = float(1)
            else:
                label_temp = float(0)
        label.ix[:, j] = label_temp
    return label

