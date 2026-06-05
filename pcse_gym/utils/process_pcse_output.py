import numpy as np


def get_previous_index(pcse_output, timestep):
    return (np.ceil(len(pcse_output) / timestep).astype('int') - 1) * timestep - 1


def get_index_for_date(pcse_output, date):
    for index in range(len(pcse_output) - 1, -1, -1):
        if pcse_output[index]['day'] == date:
            return index
    raise ValueError(f"Date {date} not found in PCSE output")


def get_start_date(pcse_output, timestep):
    return pcse_output[-1 - timestep]['day']


def get_var_names(pcse_output):
    return pcse_output[0].keys()


def get_name_storage_organ(var_names):
    # if 'TWSO' in var_names:
    #     return 'TWSO'
    if 'WSO' in var_names:
        return 'WSO'
    raise Exception(" (T)WSO not found")


def get_dict_lintul_wofost():
    dict_convert = (("WSO", "TWSO", 10.0), ("TNSOIL", "NAVAIL", 10.0))
    return dict_convert


def needs_conversion(var, dict_lintul_wofost=get_dict_lintul_wofost()):
    for lintul, wofost, conv_factor in dict_lintul_wofost:
        if var == wofost:
            return True
    return False


def get_conversion_factor(var, dict_lintul_wofost=get_dict_lintul_wofost()):
    for lintul, wofost, conv_factor in dict_lintul_wofost:
        if var == lintul:
            return conv_factor
        if var == wofost:
            return 1.0 / conv_factor
    raise Exception(f'{var} not found')


def get_n_storage_organ(pcse_output):
    return pcse_output[-1]['NamountSO'] or 0.0


def get_year_in_step(pcse_output):
    return pcse_output[-1]['day'].year


def compute_growth_var(pcse_output, timestep, var, start_date=None):
    if start_date is None:
        start_index = get_previous_index(pcse_output, timestep)
    else:
        start_index = get_index_for_date(pcse_output, start_date)
    var_start = pcse_output[start_index][var]
    var_finish = pcse_output[-1][var]
    if var_start is None or not np.isfinite(var_start):
        var_start = 0.0
    if var_finish is None or not np.isfinite(var_finish):
        var_finish = 0.0
    growth = var_finish - var_start
    return growth


def compute_growth_storage_organ(pcse_output, timestep, multiplier_amount=1, start_date=None):
    """
    Computes growth of storage organ in kg/ha if in WOFOST; g/m2 if in LINTUL
    """
    wso_var = "WSO"  # get_name_storage_organ(get_var_names(pcse_output))
    wso_growth = compute_growth_var(pcse_output, timestep, wso_var, start_date=start_date)
    wso_growth = wso_growth / multiplier_amount
    return wso_growth
