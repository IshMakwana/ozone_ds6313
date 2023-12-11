# %%
from dash import Dash, html, Input, Output, dcc
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
import dash_daq as daq

from pyrecord import Record
import sys
import os

# sys.path.append(os.getcwd() + '/../..')
sys.path.append(os.getcwd())
import toolbox as mtb

# Incorporate data
data_sets = [
    # first address records hosted on git
    'https://raw.githubusercontent.com/IshMakwana/gwu_time_series_tp/main/pollution_record1.csv',
    # whole dataset
    'pollution_2000_2022.csv']

# df = pd.read_csv(data_sets[0], parse_dates=['Date'])
pollution_df = pd.read_csv(data_sets[1], parse_dates=['Date'])

def findIthAddress(ith=1):
    i = 0
    for addy in pollution_df['Address'].unique():
        records = pollution_df.query('Address == @addy')
        if len(records) > 5000:
            if (i+1) == ith:
                return records
            i += 1


df = findIthAddress()
df.reset_index(inplace=True)
columns = df.select_dtypes(include=np.number).columns.tolist()

# data cleaning
dates = df['Date']
dr = pd.date_range(start=dates[0], end=dates[len(df)-1], freq='d')
df = mtb.driftFill(df, 'Date', dr, columns)

w_df = mtb.downSample(7, df, 'Date', columns)
std_df = mtb.standardize(df[columns])

# # Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# %%
# Data Description
DEFAULT_DEP = 'O3 AQI'
DEFAULT_IND = [
    'O3 Mean', 'O3 1st Max Value', 'O3 1st Max Hour',
    'CO Mean', 'CO 1st Max Value', 'CO 1st Max Hour', 'CO AQI',
    'SO2 Mean', 'SO2 1st Max Value', 'SO2 1st Max Hour', 'SO2 AQI',
    'NO2 Mean', 'NO2 1st Max Value', 'NO2 1st Max Hour', 'NO2 AQI'
]
# Globals
AppRecord = Record.create_type('AppRecord', 'ylabel', 'dep', 'ind', 'lags',
                               'test_size', 'alpha',
                               'seas', 'hw_seas', 'hw_trend', 'hw_damped',
                               'removed1', 'removed2', 'removed3',
                               'j', 'k', 'na', 'nb', 'arima_s', 'arima_t')

appRec = AppRecord('Timeline', DEFAULT_DEP, DEFAULT_IND, 20,
                   0.2, 0.3,
                   365, 'add', 'add', True,
                   [], [], [],
                   7, 7, 1, 2, 52, 't')
df.set_index('Date', inplace=True)

appRec.removed1, adjr2_options = mtb.backStepReg(
    std_df, appRec.dep, appRec.ind, appRec.test_size)
appRec.removed2, pval_options = mtb.backStepReg(
    std_df, appRec.dep, appRec.ind, appRec.test_size, False)
appRec.removed3, vif_options = mtb.vifMethod(
    std_df, appRec.dep, appRec.ind, appRec.test_size)


def ldiff(xcl, all):
    return [x for x in all if x not in xcl]


# App layout
app.layout = dbc.Container([
    html.H1(children='Time Series Term Project'),
    dbc.Row([
        dbc.Col([
            html.P("Dependent Variable"),
            dcc.Dropdown(
                id='dependent_variable',
                value=DEFAULT_DEP,
                clearable=False,
                options=columns)
        ]),
        dbc.Col([
            html.P("Independent Variable"),
            dcc.Dropdown(
                id='independent_variable',
                value=DEFAULT_IND,
                multi=True,
                options=columns)
        ])
    ]),
    dbc.Row([
        html.H2("Stationarity Tests"),
        dbc.Row([
            dbc.Col([
                html.Img(id='data-roll-mean-var'),
            ], width=6),
            dbc.Col([
                html.Img(id='data-acf'),
                dcc.Slider(20, 300, 20,
                           value=appRec.lags,
                           id='lags-slider'
                           ),
            ], width=6)
        ])
        # ADF & KPSS
    ]),
    dbc.Row([
        html.H2("Basic Forecasts"),
        dbc.Row([
            dbc.Button("Update Plot", color="primary",
                       className="me-1", id="show-basic-forecast",
                       style={'width': 200, 'margin-top': 20, 'margin-bottom': 20}),
            html.Img(id='data-basic-forecast'),
        ])
    ]),
    dbc.Row([
        html.H2("Holt Winters Forecast"),
        dbc.Row([
            dbc.Col([
                html.P("Seasonal"),
                dcc.Dropdown(
                    id='hw_seasonal',
                    value=appRec.hw_seas,
                    clearable=False,
                    options=['add', 'mul', None])
            ]),
            dbc.Col([
                html.P("Seasonal Periods"),
                daq.NumericInput(
                    id='hw_seasonal_periods',
                    value=appRec.seas,
                    min=0,
                    max=366,
                    size=100)
            ]),
            dbc.Col([
                html.P("Trend"),
                dcc.Dropdown(
                    id='hw_trend',
                    value=appRec.hw_trend,
                    clearable=False,
                    options=['add', 'mul', None])
            ]),
            dbc.Col([
                html.P("Damping"),
                daq.ToggleSwitch(
                    id='hw_damped',
                    value=appRec.hw_damped
                ),
            ]),
        ]),
        dbc.Row([
            dbc.Button("Update Plot", color="primary",
                       className="me-1", id="show-hw-forecast",
                       style={'width': 200, 'margin-top': 20, 'margin-bottom': 20}),
        ]),
        html.Img(id='data-hw-forecast'),
    ]),
    dbc.Row([
        html.H2("Multiple Linear Regression Models"),

        html.H4("Adjusted RSQ based Regression Model"),
        html.P("Remove features based on Adjusted RSQ. Check a feature to remove it, and update the forecast."),
        dbc.Checklist(options=adjr2_options,
                      value=appRec.removed1, inline=True,
                      id='adjr2_checklist', switch=True),
        dbc.Button("Update Plot", color="primary",
                   className="me-1", id="show-linreg-adjr2",
                   style={'width': 200, 'margin-top': 20, 'margin-bottom': 20}),
        html.Img(id='data-linreg-adjr2'),

        html.H4("P-Value based Regression Model"),
        html.P("Remove features based on P-Value. Check a feature to remove it, and update the forecast."),
        dbc.Checklist(options=pval_options,
                      value=appRec.removed2, inline=True,
                      id='pval_checklist', switch=True),
        dbc.Button("Update Plot", color="primary",
                   className="me-1", id="show-linreg-pval",
                   style={'width': 200, 'margin-top': 20, 'margin-bottom': 20}),
        html.Img(id='data-linreg-pval'),

        html.H4("VIF based Regression Model"),
        html.P(
            "Remove features based on VIF. Check a feature to remove it, and update the forecast."),
        dbc.Checklist(options=vif_options,
                      value=appRec.removed3, inline=True,
                      id='vif_checklist', switch=True),
        dbc.Button("Update Plot", color="primary",
                   className="me-1", id="show-linreg-vif",
                   style={'width': 200, 'margin-top': 20, 'margin-bottom': 20}),
        html.Img(id='data-linreg-vif'),
    ]),
    dbc.Row([
        html.H2("ARMA | ARIMA | SARIMA models"),
        dbc.Row([
            dbc.Col([
                html.P('J'),
                daq.NumericInput(id='j_value', value=7,
                                 min=7, max=12, size=100),
            ]),
            dbc.Col([
                html.P('K'),
                daq.NumericInput(id='k_value', value=7,
                                 min=7, max=12, size=100),
            ]),
        ]),
        dbc.Button("Update GPAC", color="primary",
                   className="me-1", id="show-gpac-stuff",
                   style={'width': 200, 'margin-top': 20, 'margin-bottom': 20}),
        dbc.Row([
            dbc.Col([
                html.Img(id='data-gpac-table'),
            ], width=6),
            dbc.Col([
                html.Img(id='data-acf-pacf'),
                dcc.Slider(20, 300, 20,
                           value=appRec.lags,
                           id='lags-slider2'
                           ),
            ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
                html.P('na | Na'),
                daq.NumericInput(id='na_value', value=appRec.na,
                                 min=0, max=10, size=100),
            ]),
            dbc.Col([
                html.P('nb | Nb'),
                daq.NumericInput(id='nb_value', value=appRec.nb,
                                 min=0, max=10, size=100),
            ]),
            dbc.Col([
                html.P('Seasonality'),
                daq.NumericInput(id='arima_seas', value=appRec.arima_s,
                                 min=1, max=365, size=100),
            ]),
            dbc.Col([
                html.P('Trend'),
                dcc.Dropdown(id='arima_trend', value=appRec.arima_t,
                             options=['n', 'c', 't', 'ct']),
            ]),
        ]),
        dbc.Button("Update Plot", color="primary",
                   className="me-1", id="show-arima-forecast",
                   style={'width': 200, 'margin-top': 20, 'margin-bottom': 20}),
        html.Img(id='data-arima-forecast'),
    ]),
    html.P('The End', id='no-op', style={'display': 'block'})
])

# %%

@app.callback(
    Output(component_id='adjr2_checklist', component_property='value'),
    Output(component_id='adjr2_checklist', component_property='options'),
    Output(component_id='pval_checklist', component_property='value'),
    Output(component_id='pval_checklist', component_property='options'),
    Output(component_id='vif_checklist', component_property='value'),
    Output(component_id='vif_checklist', component_property='options'),
    Input('independent_variable', 'value'),
    prevent_initial_call=True
)
def plot_data(ind):
    appRec.ind = ind
    adj_val, adj_opt = mtb.backStepReg(
        std_df, appRec.dep, appRec.ind, appRec.test_size)
    pval_val, pval_opt = mtb.backStepReg(
        std_df, appRec.dep, appRec.ind, appRec.test_size, False)
    vif_val, vif_opt = mtb.vifMethod(
        std_df, appRec.dep, appRec.ind, appRec.test_size)
    return adj_val, adj_opt, pval_val, pval_opt, vif_val, vif_opt


@app.callback(
    Output(component_id='data-roll-mean-var', component_property='src'),
    Output(component_id='data-acf', component_property='src'),
    Input('dependent_variable', 'value'),
)
def plot_data(dep):
    appRec.dep = dep
    mv = mtb.plot_rolling_mean_var(df[appRec.dep], appRec.ylabel, appRec.dep)
    acf = mtb.plotAcf(df[appRec.dep], appRec.lags, appRec.dep)
    return mv, acf


@app.callback(
    Output(component_id='data-acf',
           component_property='src', allow_duplicate=True),
    Input('lags-slider', 'value'),
    prevent_initial_call=True
)
def plot_data(lags):
    appRec.lags = lags
    acf = mtb.plotAcf(df[appRec.dep], appRec.lags, appRec.dep)
    return acf

@app.callback(
    Output(component_id='data-basic-forecast', component_property='src'),
    Input('show-basic-forecast', 'n_clicks'),
    prevent_initial_call=True
)
def plot_data(n_clicks):
    return mtb.plot_basic_forecasts(df[appRec.dep], appRec.test_size, appRec.alpha, appRec.ylabel, appRec.dep)


@app.callback(
    Output(component_id='data-hw-forecast', component_property='src'),
    Input('show-hw-forecast', 'n_clicks'),
    prevent_initial_call=True
)
def plot_data(n_clicks):
    return mtb.plotHoltWinters(df[appRec.dep], appRec.test_size,
                               trend=appRec.hw_trend, damped=appRec.hw_damped,
                               seasonal=appRec.hw_seas,
                               seasonal_periods=appRec.seas,
                               title='Holts Winters', xlabel=appRec.ylabel, ylabel=appRec.dep)


@app.callback(
    Output(component_id='no-op', component_property='children',
           allow_duplicate=True),
    [Input('hw_seasonal', 'value'),
     Input('hw_seasonal_periods', 'value'),
     Input('hw_trend', 'value'),
     Input('hw_damped', 'value')],
    prevent_initial_call=True
)
def plot_data(hw_seas, seas, hw_trend, hw_damped):
    appRec.hw_seas = hw_seas
    if appRec.hw_seas is None:
        appRec.seas = None
    else:
        appRec.seas = seas

    appRec.hw_trend = hw_trend
    if appRec.hw_trend is None:
        appRec.hw_damped = False
    else:
        appRec.hw_damped = hw_damped

    return 'The End'


@app.callback(
    Output(component_id='no-op', component_property='children',
           allow_duplicate=True),
    Input('adjr2_checklist', 'value'),
    prevent_initial_call=True
)
def plot_data(removed):
    appRec.removed1 = removed
    return 'The End'


@app.callback(
    Output(component_id='data-linreg-adjr2', component_property='src'),
    Input('show-linreg-adjr2', 'n_clicks'),
    prevent_initial_call=True
)
def plot_data(n_clicks):
    return mtb.plotRegressPrediction(df, appRec.test_size, appRec.dep,
                                     ldiff(appRec.removed1, appRec.ind), 'Adjusted R2 based Linear Regression', appRec.dep, appRec.ylabel)


@app.callback(
    Output(component_id='no-op', component_property='children',
           allow_duplicate=True),
    Input('pval_checklist', 'value'),
    prevent_initial_call=True
)
def plot_data(removed):
    appRec.removed2 = removed
    return 'The End'


@app.callback(
    Output(component_id='data-linreg-pval', component_property='src'),
    Input('show-linreg-pval', 'n_clicks'),
    prevent_initial_call=True
)
def plot_data(n_clicks):
    return mtb.plotRegressPrediction(df, appRec.test_size, appRec.dep,
                                     ldiff(appRec.removed2, appRec.ind), 'P-Value based Linear Regression', appRec.dep, appRec.ylabel)


@app.callback(
    Output(component_id='no-op', component_property='children',
           allow_duplicate=True),
    Input('vif_checklist', 'value'),
    prevent_initial_call=True
)
def plot_data(removed):
    appRec.removed3 = removed
    return 'The End'


@app.callback(
    Output(component_id='data-linreg-vif', component_property='src'),
    Input('show-linreg-vif', 'n_clicks'),
    prevent_initial_call=True
)
def plot_data(n_clicks):
    return mtb.plotRegressPrediction(df, appRec.test_size, appRec.dep,
                                     ldiff(appRec.removed3, appRec.ind), 'VIF based Linear Regression', appRec.dep, appRec.ylabel)

@app.callback(
    Output('no-op', 'children', allow_duplicate=True),
    [Input('j_value', 'value'), Input('k_value', 'value')],
    prevent_initial_call=True
)
def save_date(j, k):
    appRec.j = j
    appRec.k = k
    return 'The End'


@app.callback(
    Output('no-op', 'children', allow_duplicate=True),
    [Input('na_value', 'value'), Input('na_value', 'value'),
     Input('arima_seas', 'value'), Input('arima_trend', 'value')],
    prevent_initial_call=True
)
def save_date(na, nb, arima_s, arima_t):
    appRec.na = na
    appRec.nb = nb
    appRec.arima_s = arima_s
    appRec.arima_t = arima_t
    print('updated arimas')
    return arima_t

@app.callback(
    Output(component_id='data-gpac-table', component_property='src'),
    Output(component_id='data-acf-pacf', component_property='src'),
    Input('show-gpac-stuff', 'n_clicks'),
    prevent_initial_call=True
)
def show_table(n):
    y = w_df[appRec.dep].values.flatten()
    gpac = mtb.showGPAC(y, appRec.j, appRec.k)
    apacf = mtb.ACF_PACF_Plot(y, appRec.lags)
    return gpac, apacf

@app.callback(
    Output(component_id='data-acf-pacf',
           component_property='src', allow_duplicate=True),
    Input('lags-slider2', 'value'),
    prevent_initial_call=True
)
def plot_data(lags):
    appRec.lags = lags
    y = w_df[appRec.dep].values.flatten()
    apacf = mtb.ACF_PACF_Plot(y, appRec.lags)
    return apacf

@app.callback(
    Output(component_id='data-arima-forecast', component_property='src'),
    Input('show-arima-forecast', 'n_clicks'),
    prevent_initial_call=True
)
def show_plot(n):
    order = (0, 0, 0)
    s_order = (0, 0, 0, 0)
    if appRec.arima_s > 1:
        s_order = (appRec.na, 0, appRec.nb, appRec.arima_s)
    else:
        order = (appRec.na, 0, appRec.nb)
    return mtb.plotARIMA(w_df[appRec.dep],
                         appRec.test_size,
                         order,
                         appRec.arima_t,
                         s_order,
                         appRec.dep,
                         appRec.ylabel)

# %%
# Run the app
if __name__ == '__main__':
    app.run(debug=True)


