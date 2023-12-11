import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import statsmodels.tsa.holtwinters as ets
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
matplotlib.use('agg')


def_fs = (6, 6)
wide_fs = (12, 6)
DEC = 3

J = 7
K = 7

def standardize(x): return (x - x.mean()) / x.std()

def encodedFig(l=False, g=False):
    if l:
        plt.legend()
    if g:
        plt.grid()
    fig = plt.gcf()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.clf()
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f'data:image/png;base64,{fig_data}'


def plotXY(X, Y, title, xlabel, ylabel, show=True, fs=(9, 9)):
    if fs != None:
        plt.figure(figsize=fs)
    plt.plot(X, Y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    if show:
        # plt.show()
        return encodedFig()

    return None


def plot_rolling_mean_var(data, xlabel, ylabel):
    rolling_means = []
    rolling_variances = []
    for i in range(len(data)):
        rolling_means.append(np.mean(data[:i]))
        rolling_variances.append(np.var(data[:i]))
    # plot
    x = data.index
    plt.figure(figsize=def_fs)
    plt.subplot(2, 1, 1)
    plotXY(x, rolling_means,
           xlabel=xlabel, ylabel=ylabel, title='Rolling Mean',
           show=False, fs=None)
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plotXY(x, rolling_variances,
           title='Rolling Variance', xlabel=xlabel, ylabel=ylabel,
           show=False, fs=None)
    plt.tight_layout()
    # plt.show()
    return encodedFig()


def mkDblSd(dl, lags):
    return range(-lags, lags+1), (dl[::-1] + dl[1:])


def acf(df, lags):
    ybar = df.mean()
    den = np.square(df - ybar).sum()
    T = len(df)

    res = []
    for r in range(lags + 1):
        nom = (df[r:] - ybar) * (df[:T - r] - ybar)
        res.append(nom.sum() / den)

    return res


def plotAcf(data, num_lags, title):
    # Plot ACF
    sig = 1.96 / np.sqrt(len(data))

    flattened_data = np.array(data).flatten()
    acf_val = acf(flattened_data, num_lags)
    x_a, y_a = mkDblSd(acf_val, num_lags)
    plt.figure(figsize=def_fs)
    (m, s, b) = plt.stem(x_a, y_a)
    plt.setp(m, color='red')
    plt.axhspan(-sig, sig, alpha=0.3, color='blue')
    plt.title(f'ACF of {title}, #lags={num_lags}')
    plt.xlabel('#Lags')
    plt.ylabel('Magnitude')

    # plt.show()
    return encodedFig()


def plotForecast(train, test, forecast, method, xlabel, ylabel, show=True):
    # plt.figure(figsize=(12,4))
    plt.plot(train.index, train, label='Training Set')
    plt.plot(test.index, test, label='Test Set')
    plt.plot(test.index, forecast, label='Forecast')

    plt.title(f'{method} Forecast')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if show:
        # plt.show()
        return encodedFig()

# Basic Prediction Models


def getAvgModel(trn, tst):
    mean = trn.mean()

    hSteps = [trn[0]]
    for i in range(1, len(trn)):
        hSteps.append(trn[:i].mean())

    return np.full(shape=(len(tst)), fill_value=mean), np.array(hSteps)


def getNaiveModel(trn, tst):
    hSteps = [trn[0]]
    for i in range(1, len(trn)):
        hSteps.append(trn[i-1])

    return np.full(shape=(len(tst)), fill_value=trn[-1]), np.array(hSteps)


def getDriftModel(trn, tst):
    hSteps = [trn[0], trn[1]]
    for i in range(2, len(trn)):
        step = trn[i - 1] + (trn[i - 1] - trn[0]) / (i - 1)
        hSteps.append(step)

    forecasts = []
    increment = (trn[-1] - trn[0]) / (len(trn) - 1)
    for i in range(len(tst)):
        forecasts.append(trn[-1] + (i+1) * increment)

    return np.array(forecasts), np.array(hSteps)


def getSESModel(trn, tst, alpha=0.5):
    tSteps = [trn[0]]
    for i in range(1, len(trn)):
        step = alpha * trn[i - 1] + (1 - alpha) * tSteps[i-1]
        tSteps.append(step)

    frcst = alpha * trn[-1] + (1 - alpha) * tSteps[-1]

    return np.full(shape=(len(tst)), fill_value=frcst), np.array(tSteps)


def plot_basic_forecasts(data, test_size, alpha, xlabel, ylabel):
    train, test = train_test_split(data, test_size=test_size, shuffle=False)
    forecast, steps = getAvgModel(train, test)
    plt.figure(figsize=(12, 16))
    plt.subplot(4, 1, 1)
    plotForecast(train, test, forecast, 'Average', xlabel, ylabel, show=False)

    forecast, steps = getNaiveModel(train, test)
    plt.subplot(4, 1, 2)
    plotForecast(train, test, forecast, 'Naive', xlabel, ylabel, show=False)

    forecast, steps = getDriftModel(train, test)
    plt.subplot(4, 1, 3)
    plotForecast(train, test, forecast, 'Drift', xlabel, ylabel, show=False)

    forecast, steps = getSESModel(train, test, alpha)
    plt.subplot(4, 1, 4)
    plotForecast(train, test, forecast,
                 f'SES a={alpha}', xlabel, ylabel, show=False)
    return encodedFig()


def predictHoltWinters(train, test, trend=None, damped=False, seasonal=None, seasonal_periods=None):
    holtt = ets.ExponentialSmoothing(
        train, trend=trend, damped_trend=damped, seasonal=seasonal, seasonal_periods=seasonal_periods).fit()
    return holtt.forecast(steps=len(test))

# plot holt winters.
# trend = 'mul' | 'add' | None,
# damped = True | False, only provide if trend is not None
# seasonal = 'mul' | 'add' | None,
# seasonal_periods is a number


def plotHoltWinters(data, test_size, trend=None, damped=False,
                    seasonal=None,
                    seasonal_periods=None,
                    title='Holts Winters', xlabel='Date', ylabel='Data'):
    train, test = train_test_split(data, test_size=test_size, shuffle=False)
    holtf = predictHoltWinters(
        train, test, trend, damped, seasonal, seasonal_periods)
    holtf = pd.DataFrame(holtf).set_index(test.index)

    # forecastAccuracy(test.values, holtf.values, 'Holt-Winters')
    plt.figure(figsize=wide_fs)
    return plotForecast(train, test, holtf, 'Holt-Winters', xlabel, ylabel)


def backStepReg(data, dep, feats, test_size, use_adjr2=True):
    train, _ = train_test_split(data, test_size=test_size, shuffle=False)
    Y = np.array(train[dep])
    X = np.array(train[feats])

    model = sm.OLS(Y, sm.add_constant(X)).fit()
    removed = []

    pval_df = pd.DataFrame()
    pval_df['pvals'] = model.pvalues[1:]
    pval_df['features'] = feats

    # print('Features:')
    # print(np.array(feats))
    analysis = []
    # analysis.append(f'Adjusted_R2: {model.rsquared_adj:.4f} initially')
    sorted_decreasing = pval_df.sort_values('pvals', ascending=False)
    # if use_adjr2 else sorted_decreasing['pvals'][0]
    prev = model.rsquared_adj

    for i, col in enumerate(sorted_decreasing['features']):
        cols = [x for x in sorted_decreasing['features']
                if x not in (removed + [col])]
        pval = sorted_decreasing['pvals'][i]
        model = sm.OLS(Y, sm.add_constant(train[cols].to_numpy())).fit()

        if use_adjr2 and model.rsquared_adj >= prev:
            removed.append(col)
        elif not (use_adjr2) and pval > 0.05:
            removed.append(col)

        str = f'[Adjusted_R2: {model.rsquared_adj:.3f}]' if use_adjr2 else f'[P-Value: {pval:.3f}]'
        analysis.append({'label': f'{col} - {str}', 'value': col})

    return removed, analysis

# Regression Models - 2. VIF Analysis


def vifMethod(data, dep, feats, test_size, threshold=10):
    train, _ = train_test_split(data, test_size=test_size, shuffle=False)
    X = sm.add_constant(np.array(train[feats]))
    Y = np.array(train[dep])

    model = sm.OLS(Y, X).fit()
    removed = []

    vif = pd.DataFrame()
    vif['columns'] = ['ones']+feats
    vif['vif'] = [variance_inflation_factor(
        X, i) for i in range(len(vif['columns']))]
    sorted_decreasing = vif.sort_values('vif', ascending=False)

    # print('Question 8 Solution:')
    # print(f'{sorted_decreasing}\n')
    analysis = []
    recommend = []

    # analysis.append(
    #     f'Adjusted_R2: {model.rsquared_adj:.4f} before removing')

    for i, col in enumerate(sorted_decreasing['columns']):
        cols = [x for x in feats if x not in (removed + [col])]
        vif_v = sorted_decreasing['vif'][i]
        model = sm.OLS(Y, sm.add_constant(np.array(train[cols]))).fit()

        if vif_v > threshold and col != 'ones':
            recommend.append(col)
        removed.append(col)
        if col != 'ones':
            analysis.append({'label': f'{col} - [vif: {vif_v:.3f}]', 'value': col})

    return recommend, analysis


def linRegPredict(train, test, dep, feats):
    model = sm.OLS(train[dep].to_numpy(),
                   sm.add_constant(
        train[feats].to_numpy())).fit()
    return model, model.predict(sm.add_constant(
        test[feats].to_numpy()))


def plotRegressPrediction(data, test_size, dep, feats, title, xlabel, ylabel):
    train, test = train_test_split(data, test_size=test_size, shuffle=False)

    _, prediction = linRegPredict(train, test, dep, feats)
    # prediction = pd.DataFrame(prediction).set_index(test.index)

    plt.figure(figsize=wide_fs)
    return plotForecast(train[dep], test[dep], prediction, title, xlabel, ylabel)

# GPAC Table generation


def fi_j_k(ry, j, k):
    num = 1
    den = 1

    if k == 1:
        num = ry[j + 1]
        den = ry[j - k + 1]
    else:
        mat = []
        for i in range(k):
            mat.append([])
            for l in range(k-1):
                ind = np.absolute(j - l + i)
                mat[i].append(ry[ind])
            mat[i].append(ry[j + 1 + i])

        mat = np.array(mat)
        # print(f'numerator matrix: {mat}')
        num = np.linalg.det(mat)

        mat = []
        for i in range(k):
            mat.append([])
            for l in range(k):
                ind = np.absolute(j - l + i)
                mat[i].append(ry[ind])

        mat = np.array(mat)
        # print(f'denominator matrix: {mat}')
        den = np.linalg.det(mat)

    # if num == 0:
    #     return 0.0
    return (num / den)


def showGPAC(data, j=J, k=K):
    ry = acf(data, j+k+1)
    table = []
    # print(ry)
    for i in range(j):
        table.append([])
        for l in range(1, k):
            val = fi_j_k(ry, i, l)
            table[i].append(val)

    table = pd.DataFrame(table, columns=list(range(1, k)))
    plt.figure(figsize=(6, 6))
    sns.heatmap(table, annot=True, cmap='magma', fmt=f'.{DEC}f')
    plt.title("GPAC Table")
    return encodedFig()


def ACF_PACF_Plot(y, lags):
    # acf = sm.tsa.stattools.acf(y, nlags=lags)
    # pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    # plt.show()
    return encodedFig()


def plotARIMA(data, test_size, order=(0, 0, 0), trend='n', seasonal=(0, 0, 0, 0), xlabel='', ylabel=''):
    # STEP 9
    train, test = train_test_split(data, test_size=test_size, shuffle=False)
    model = sm.tsa.arima.ARIMA(
        train, order=order, trend=trend, seasonal_order=seasonal).fit()
    model_hat = model.forecast(len(test))
    plt.figure(figsize=wide_fs)
    return plotForecast(train, test, model_hat, 'ARIMA', xlabel, ylabel)


def driftFill(data, old_index_column, new_index, columns):
    new_index = list(new_index)
    result = {}
    result[old_index_column] = new_index
    old_index = {}

    for i, v in enumerate(data[old_index_column]):
        old_index[v] = i

    for c in columns:
        initial = data[c][0]
        new_data = []
        for i, j in enumerate(new_index):
            if j in old_index:
                x = old_index[j]
                new_data.append(data[c][x])
            else:
                if i > 1:
                    new_item = new_data[i - 1] + \
                        (new_data[i - 1] - new_data[0]) / (i - 1)
                    new_data.append(new_item)
                else:
                    new_data.append(initial)

        result[c] = new_data
    return pd.DataFrame(result)


def downSample(step, data, index, columns):
    ds_data = pd.DataFrame(columns=columns)
    total = len(data)
    dates = []
    for i in range(step, total, step):
        ds_data.loc[len(ds_data.index)] = data[columns][i-step:i].mean()
        dates.append(data[index][i])
    ds_data[index] = dates
    return ds_data
