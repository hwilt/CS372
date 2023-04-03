import numpy as np
import matplotlib.pyplot as plt
from novfn import *
from tempo import *

def get_gt_beats(filename, anno=0):
    """
    Load in the ground truth beats for some clip
    
    Parameters
    ----------
    filename: string
        Path to annotations
    anno: int
        Annotator number
    """
    fin = open(filename)
    lines = fin.readlines()
    fin.close()
    beats = lines[anno]
    beats = np.array([float(x) for x in beats.split()])
    return beats

def plot_beats(novfn, beats, sr, hop_length):
    """
    Plot the location of beats superimposed on an audio novelty
    function

    Parameters
    ----------
    novfn: ndarray(N)
        An audio novelty function
    beats: ndarray(M)
        An array of beat locations, in seconds
    sr: int
        Sample rate
    hop_length: int
        Hop length used in the STFT to construct novfn
    """
    h1 = np.min(novfn)
    h2 = np.max(novfn)
    diff = (h2-h1)
    h2 += 0.3*diff
    h1 -= 0.3*diff
    for b in beats:
        plt.plot([b, b], [h1, h2], linewidth=1)
    ts = np.arange(novfn.size)
    plt.plot(ts*hop_length/sr, novfn, c='C0', linewidth=2)
    plt.xlabel("Time (Sec)")


def sonify_beats(x, sr, beats, blip_len=0.03):
    """
    Put short little 440hz blips at each beat location
    in an audio file

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate of audio
    beats: ndarray(N)
        Beat locations, in seconds, of each beat
    blip_len: float 
        The length, in seconds, of each 440hz cosine blip
    """
    y = np.array(x) # Copy x into an array where the beats will be sonified
    ## TODO: Fill this in and add the blips
    for b in beats:
        start = int(b*sr)
        end = int((b+blip_len)*sr)
        y[start:end] += np.cos(2*np.pi*440*np.arange(end-start)/sr)
    return y

def get_beats(novfn, sr, hop_length, tempo, alpha):
    """
    An implementation of dynamic programming beat tracking

    Parameters
    ----------
    novfn: ndarray(N)
        An audio novelty function
    sr: int
        Sample rate
    hop_length: int
        Hop length used in the STFT to construct novfn
    tempo: float
        The estimated tempo, in beats per minute
    alpha: float
        The penalty for tempo deviation
    
    Returns
    -------
    ndarray(B): 
        Beat locations, in seconds, of each beat
    """
    N = len(novfn)
    cscore = np.array(novfn) # Dynamic programming array
    backlink = np.ones(N, dtype=int) # Links for backtracing
    T = int((60*sr/hop_length)/tempo) # Period, in units of novfn samples per beat, of the tempo
    beats = []

    ## TODO: Fill this in
    #print(T)
    ## cscore[i] = max(novfn[i] + cscore[i-T/2] - alpha|log((i-j)/T)|^2, novfn[i] + cscore[i-2*T] - alpha|log((i-j)/T)|^2)
    #for i in range(T, N):
    #    cscore[i] = max(novfn[i] + cscore[i-T//2] - alpha*((np.log((i-T//2)/T))**2), novfn[i] + cscore[i-2*T] - alpha*((np.log((i-2*T)/T)))**2)
    #    if cscore[i] == novfn[i] + cscore[i-T//2] - alpha*(np.abs(np.log((i-T//2)/T))**2):
    #        backlink[i] = i-T//2
    #    else:
    #        backlink[i] = i-2*T

    for i in range(1,N):
        if i-T//2 > 0:
            cscore[i] = cscore[i-2*T] + alpha
        else:
            #print(i-T)
            cscore[i] = np.min(cscore[i-T:i]) + alpha
            backlink[i] = np.argmin(cscore[i-T:i]) + i-T

    # plot the cscore
    """plt.figure()
    plt.plot(cscore)
    plt.xlim(0, 400)
    plt.show()"""

    # Backtrace
    i = N-1
    while i > 0:
        beats.append(i)
        i = backlink[i]
    beats = np.array(beats)
    beats = beats[::-1]
    beats = beats*hop_length/sr

   


    return beats