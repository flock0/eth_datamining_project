import numpy as np
from numpy.random import RandomState
import math
from time import sleep

mean = np.asarray([0.00212085,0.00329081,0.00334493,0.00174832,0.00346658,0.00363884,0.0019232,0.00218578,0.0018738,0.00345416,0.00527891,0.00451388,0.00352587,0.00304566,0.0022496,0.00178546,0.00215941,0.00378952,0.00257443,0.00292485,0.00616323,0.0023531,0.00439119,0.00719164,0.00371757,0.00057973,0.00302755,0.00631249,0.001839,0.00181765,0.00327509,0.00351156,0.00246338,0.00262781,0.00355964,0.00305285,0.00214029,0.00124059,0.00212686,0.00301145,0.00161364,0.01254419,0.00251699,0.0052467,0.00265408,0.00282775,0.00392861,0.00207452,0.00308604,0.00202386,0.00407048,0.00184065,0.00319065,0.00349774,0.00410208,0.00345629,0.00070687,0.00130379,0.00237483,0.00395205,0.00244048,0.01222654,0.00125193,0.00267496,0.00432702,0.00065104,0.00222167,0.00080984,0.00277268,0.0011927,0.00331095,0.00336796,0.00265644,0.00097431,0.00217861,0.00262219,0.00157242,0.00302583,0.00409985,0.0027509,0.00079443,0.00260117,0.00272446,0.00279359,0.00186658,0.0024002,0.00320262,0.00289812,0.00126004,0.00220894,0.00216796,0.00227115,0.00366024,0.00280223,0.00058087,0.00307912,0.00279961,0.00131907,0.00075948,0.00360067,0.00327285,0.0023322,0.00317545,0.0038663,0.00270042,0.00097353,0.00230831,0.00136515,0.00457633,0.00384655,0.00043474,0.00152411,0.00247713,0.00180558,0.00285459,0.0005569,0.00417167,0.00232106,0.00358048,0.00194744,0.00222136,0.00213187,0.0014126,0.00608418,0.00210237,0.00230699,0.00121303,0.00296537,0.00231132,0.00087045,0.00352865,0.00227238,0.00364882,0.00272483,0.00197321,0.00207677,0.0014703,0.00017216,0.0022452,0.00351829,0.00306054,0.00237375,0.00042953,0.00300329,0.003716,0.00242389,0.00086272,0.00112311,0.00278677,0.00137649,0.00430663,0.00319966,0.00257313,0.00234901,0.00213248,0.00290098,0.00171351,0.00281955,0.00304834,0.00318478,0.00413332,0.00057373,0.00303551,0.00028611,0.00258634,0.00198062,0.00267306,0.00018674,0.00231064,0.00192323,0.00358222,0.0030322,0.00059151,0.00113047,0.00360398,0.0009674,0.00186979,0.00284991,0.0040155,0.0004155,0.00088867,0.00279111,0.00216259,0.00320792,0.00128694,0.00117511,0.0024484,0.00240532,0.00358276,0.0031617,0.00533721,0.00319791,0.00301377,0.00025419,0.00228907,0.0022915,0.00413729,0.00236341,0.00477765,0.00051874,0.00093649,0.00036575,0.00079013,0.00073831,0.00282323,0.00356221,0.0057396,0.00318299,0.00141891,0.00401395,0.00273739,0.00228698,0.00131094,0.00310903,0.00148064,0.0019079,0.00238803,0.0025793,0.00187686,0.00226158,0.00419605,0.00236949,0.00227139,0.00294995,0.00079681,0.00017411,0.00071746,0.00220946,0.00235196,0.00034329,0.0030057,0.00406415,0.00180515,0.00451529,0.00256088,0.00217466,0.00309889,0.00137677,0.00111755,0.00354409,0.00251799,0.00426961,0.0009102,0.00153481,0.00043354,0.00205552,0.00297062,0.00059376,0.00060024,0.00554018,0.00060417,0.00240036,0.00516953,0.00321521,0.00168634,0.00243221,0.00331923,0.00181337,0.00035981,0.00229823,0.00422696,0.00334789,0.00409729,0.00090567,0.00277889,0.00209744,0.00223239,0.0021575,0.00334717,0.00255804,0.00417758,0.00238542,0.00130148,0.00385234,0.00215963,0.00325692,0.00250227,0.00390508,0.00319618,0.00352319,0.0022784,0.00208739,0.00128177,0.00296079,0.00303143,0.00314718,0.00074212,0.00230456,0.00326084,0.00282652,0.00337842,0.00216192,0.00136735,0.00253702,0.00046989,0.00142179,0.0017771,0.00041912,0.00072709,0.00302937,0.00139263,0.00246249,0.00024798,0.00431116,0.00452686,0.00473716,0.00101888,0.00277703,0.00288935,0.00249314,0.00179183,0.00280047,0.00080775,0.00141873,0.00185381,0.00185994,0.00275027,0.00315311,0.0019266,0.00176007,0.00291326,0.00051859,0.00415036,0.00133048,0.0025186,0.00320395,0.00158798,0.0033038,0.00323432,0.00160849,0.00041696,0.00159861,0.00140099,0.0022161,0.00295134,0.00286418,0.00125491,0.00281816,0.00017974,0.00145718,0.00258574,0.00362248,0.00282088,0.00320894,0.00093376,0.00229472,0.0031169,0.00890786,0.00197341,0.00160105,0.0030537,0.00175409,0.00254137,0.00140346,0.00050445,0.00242729,0.00165663,0.00142772,0.00259018,0.00154881,0.00194417,0.00434929,0.00249317,0.00246407,0.00325768,0.00082088,0.00047807,0.00262654,0.00298718,0.00333407,0.00303987,0.00321357,0.00256124,0.00744777,0.00261531,0.00103862,0.00185216,0.00320413,0.00255425,0.00110427,0.00164903,0.00225808,0.00063944,0.0026718,0.00115647,0.00207329,0.00062871,0.00129837,0.0030746,0.00132051,0.0017368,0.00279598,0.00457747,0.00153843,0.00267897,0.00209474,0.00321403,0.00029277,0.00115719,0.00318635])
std = np.asarray([0.00196373,0.00361184,0.00353405,0.0022552,0.00432286,0.00792339,0.00354797,0.00179092,0.00255128,0.00338437,0.00520132,0.0038434,0.00580458,0.00260912,0.00266876,0.00255287,0.00194955,0.00374434,0.00306186,0.00254764,0.03564714,0.00223322,0.00355945,0.00857026,0.00309572,0.00196656,0.00648423,0.01069253,0.00160111,0.00179711,0.00342498,0.00745415,0.00190426,0.00206899,0.00399455,0.00294803,0.00206167,0.00209959,0.00242613,0.00496975,0.00520669,0.01391432,0.00226726,0.00590201,0.00237594,0.00223578,0.00485823,0.00187754,0.00222264,0.00166387,0.00338298,0.00215249,0.0028664,0.00408354,0.00424775,0.00391327,0.0009878,0.00202845,0.00224395,0.00285778,0.00249457,0.01456723,0.0015599,0.00318233,0.00399458,0.00115215,0.00197486,0.00192187,0.00213549,0.00147517,0.00380858,0.00349618,0.00429984,0.00130067,0.00192195,0.00254293,0.0028361,0.00308399,0.00377279,0.00244724,0.00135557,0.00327753,0.00234962,0.00228249,0.00161353,0.00222508,0.0036292,0.00226814,0.0014966,0.00189059,0.00200891,0.00185251,0.00348569,0.00348902,0.00115561,0.00397114,0.00224769,0.00190508,0.0021086,0.00367043,0.00286075,0.00230555,0.00276952,0.00510016,0.00214877,0.00185732,0.00204373,0.00231666,0.00642546,0.00426209,0.00121029,0.00295903,0.00196805,0.00172154,0.00266745,0.00177683,0.00305534,0.00301825,0.0040307,0.0027228,0.00253075,0.00213219,0.00209458,0.01006357,0.001956,0.00179902,0.00166137,0.00370375,0.00208522,0.00116261,0.00351486,0.00212267,0.00681193,0.00212465,0.00197721,0.00188429,0.00156366,0.00095062,0.00236019,0.00375868,0.00263253,0.0028355,0.00112545,0.00235198,0.00326389,0.00229673,0.00125466,0.00201808,0.00342377,0.00180181,0.00608163,0.00365307,0.00279547,0.00228516,0.00197072,0.00224082,0.00285097,0.00259113,0.00292102,0.00457766,0.00451042,0.0020972,0.00266799,0.0008746,0.00316926,0.00204805,0.00213767,0.0011031,0.00274814,0.0021508,0.00702969,0.00239404,0.0012201,0.00242963,0.00391858,0.00147255,0.00204122,0.00225944,0.00443631,0.00136385,0.00175905,0.00254393,0.0032124,0.00629065,0.00195719,0.00201603,0.00282652,0.0021719,0.00293236,0.00282156,0.00537125,0.00362031,0.00256891,0.00121427,0.00198416,0.0019599,0.0059853,0.00246239,0.0059585,0.00186837,0.00144533,0.00103183,0.00241517,0.00138369,0.00221349,0.0037931,0.00698082,0.00311398,0.00140581,0.0049529,0.00254017,0.0019502,0.00225457,0.00333899,0.00158544,0.00280094,0.00281893,0.00206341,0.00223586,0.0027439,0.0030242,0.00218517,0.00237951,0.00231054,0.0011241,0.00102426,0.00204652,0.00200652,0.00208569,0.00121232,0.00230293,0.00558661,0.00170612,0.00558467,0.0020794,0.00204624,0.00229559,0.00137314,0.00203056,0.00262278,0.0026662,0.00497367,0.00240725,0.00190872,0.00109287,0.0019679,0.00358239,0.00116492,0.00130754,0.01296843,0.00117738,0.00267082,0.00521511,0.00233107,0.00248339,0.00238337,0.00661885,0.00299148,0.00103214,0.00200947,0.0054332,0.00307228,0.004586,0.00122547,0.00459063,0.00170903,0.00202662,0.0019841,0.00319826,0.00339163,0.00455273,0.00218355,0.00155034,0.00497182,0.00243158,0.00264949,0.00293463,0.00495444,0.00255532,0.00377145,0.0020066,0.00274466,0.00132924,0.0048505,0.00245381,0.00547889,0.00215532,0.00205891,0.00350458,0.0023862,0.00349194,0.00280214,0.00206075,0.00205413,0.00118677,0.00149003,0.00159484,0.00122493,0.00109005,0.00297148,0.00133967,0.00295199,0.00121909,0.00550574,0.00539991,0.00401454,0.00148483,0.00302896,0.00361318,0.00235735,0.00180086,0.00279249,0.00120904,0.00135649,0.00177772,0.00206956,0.00261225,0.00268386,0.00213832,0.0020013,0.00373256,0.00123281,0.00616862,0.00217171,0.00209291,0.00407373,0.00230553,0.00291968,0.00319559,0.00210067,0.00168553,0.00157568,0.00245781,0.00189828,0.00235759,0.00404792,0.00190861,0.0027255,0.00102665,0.00251291,0.00202935,0.003888,0.00280965,0.00256449,0.00151982,0.00261572,0.00248668,0.02311915,0.00166255,0.00226684,0.0030611,0.00182701,0.00225784,0.00212698,0.00159933,0.00227986,0.00144933,0.00260931,0.00247964,0.00460098,0.00200251,0.00551382,0.00376504,0.00411697,0.0042512,0.00120022,0.00104569,0.0032628,0.00269923,0.00291444,0.00352687,0.00332514,0.00206045,0.00607698,0.00328992,0.00123999,0.00165734,0.00268707,0.00225209,0.00164237,0.00169868,0.00187539,0.00115765,0.00228161,0.00151962,0.00203904,0.00135921,0.00249386,0.00263263,0.00159578,0.00209341,0.00246419,0.00679129,0.00167572,0.00217293,0.00178056,0.00308025,0.00090932,0.00161063,0.00246275])

def transform(X_input):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    
    # Hyperparameters
    m = 2000
    gamma = 80 # 20 unten 40 oben, 80 optimum

    # Normalize the data
    #X = (X_input - mean) * (1/std)
    X = X_input

    d = 0
    if len(X.shape)<=1:
        d = len(X)
    else:
        d = X.shape[1]

    # Draw iid m samples omega from p and b from [0,2pi]
    random_state = RandomState(124)

    omega = np.sqrt(2.0 * gamma) * random_state.normal(size=(d, m))
    b = random_state.uniform(0, 2 * np.pi, size=m)

    # Transform the input
    projection = np.dot(X, omega) + b
    Z = np.sqrt(2.0/m) * np.cos(projection)

    return Z

def parseValue(value):
    arrayList = []
    for item in value:
        array = np.fromstring(item, dtype=float, sep=' ')
        arrayList.append(array)
    matrix = np.asarray(arrayList)
    X = matrix[:,1:]
    X_trans = transform(X)
    Y = matrix[:,0]
    return X_trans,Y

def calculateGradient(X, Y, lamda, w, batchsize, dimension):
    index = np.random.choice(X.shape[0], size = batchsize, replace=False)
    gradient = np.zeros(dimension)
    counter = 0

    for i in index:
        # Get the row
        x = np.ravel(X[i,:])
        y = Y[i]
        # Check if it is classified correctly with w
        # If not, add the negative direction to the gradient
        #print("check", y*w.dot(x))
        if y*w.dot(x) < 1:
            gradient = gradient - y*x
            counter = counter + 1
    print("Counter",counter)

    return gradient



def mapper(key, value):
    '''key: None
    value: one line of input file
    Implements the PEGASOS method (complementary to our EVA method)
    '''

    # Hyper parameters
    lamda = 1e-4 # unten
    T = 2000
    batchsize = 1024
    alpha = 0.2

    # Parse the input and shuffle it
    X,Y = parseValue(value)
    # TODO: shuffle the input

    # Get the dimension and samplesize
    d = X.shape[1]
    n = X.shape[0]

    # Initialize m_0, v_0, w_0 to zero
    w = np.zeros(d)
    gradient_old = np.zeros(d)

    for t in range(1,T):
        # Calculate the gradient g_t
        gradient = calculateGradient(X,Y,lamda,w,batchsize,d)
        gradient = alpha*gradient_old + (1-alpha) * gradient
        # Update the weight vector
        w = (1-(1/float(t))) * w - ((1/(lamda*float(t)))/batchsize) * gradient
        w = min(1, ((1/np.sqrt(lamda))/np.linalg.norm(w))) * w
        #sprint("new weight vector:", w)
        gradient_old = gradient
    yield "key", w  # This is how you yield a key, value pair


def reducer(key, values):
    '''
    key: key from mapper used to aggregate
    values: list of all value for that key
    Implements the EVA method (EVArage method, complementary to our ADAM)
    '''
    avg = np.average(values, axis=0)
    yield avg