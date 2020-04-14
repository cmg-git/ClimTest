#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:22:02 2020

@author: cghiaus
"""
import numpy as np
import pandas as pd
import psychro as psy
import matplotlib.pyplot as plt

# global variables
UA = 935.83                 # bldg conductance
tIsp, wIsp = 18, 6.22e-3    # indoor conditions

# constants
c = 1e3                     # air specific heat J/kg K
l = 2496e3                  # latent heat J/kg

# *****************************************
# ALL OUT AIR
# *****************************************


def ModelAllOutAir(m, tS, mi, tO, phiO):
    """
    Model:
        All outdoor air
        CAV Constant Air Volume:
            mass flow rate given
            control of indoor condition (t2, w2)
    INPUTS:
        m     mass flow of supply dry air
        tS    supply air;                30°C for design conditions
        mi    infiltration massflow rate 2.18 kg/s for design conditions
        tO    outdoor temperature;       -1°C for design conditions
        phiO  outdoor relative humidity; 100% for design conditions

    OUTPUTS:
        x     vector 10 elements:
            t0, w0, t1, w1, t2, w2, QsHC, QlVH, QsTZ, QlTZ

    System:
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions
        0, 1, 2 unknown points (temperature, humidity ratio)

    --o->HC--0->VH--1->TZ--2-->
         |       |     ||  |
         |       |     BL  |
         |       |         |
         |       |<----Kw--|-w2
         |<------------Kt--|-t2
 
    """
    Kt, Kw = 1e10, 1e10             # controller gain
    wO = psy.w(tO, phiO)           # outdoor mumidity ratio

    # Model
    A = np.zeros((10, 10))          # coefficents of unknowns
    b = np.zeros(10)                # vector of inputs

    A[0, 0], A[0, 6],           b[0] = m*c,     -1,     m*c*tO      # S heater
    A[1, 1],                    b[1] = m*l,     m*l*wO

    A[2, 0], A[2, 2],           b[2] = -m*c,    m*c,    0           # L humidif
    A[3, 1], A[3, 3], A[3, 7],  b[3] = -m*l,    m*l,    -1,     0

    A[4, 2], A[4, 4], A[4, 8],  b[4] = -m*c,    m*c,    -1,     0   # Z zone
    A[5, 3], A[5, 5], A[5, 9],  b[5] = -m*l,    m*l,    -1,     0

    A[6, 4], A[6, 8],           b[6] = UA+mi*c, 1,  (UA + mi*c)*tO  # Bldg
    A[7, 5], A[7, 9],           b[7] = mi*l,    1,  mi*l*wO

    A[8, 4], A[8, 8],           b[8] = Kt,      1,  Kt*tIsp         # Controler

    A[9, 5], A[9, 9],           b[9] = Kw,      1,  Kw*wIsp         # Controler

    # Solution
    x = np.linalg.solve(A, b)
    return x


def AllOutAirCAV(tS=30, mi=2.18, tO=-1, phiO=1):
    """
    All out air
    CAV Constant Air Volume:
        mass flow rate calculated for design conditions
        maintained constant in all situations

    INPUTS:
        tS    supply air;                30°C for design conditions
        mi    infiltration massflow rate 2.18 kg/s for design conditions
        tO    outdoor temperature;       -1°C for design conditions
        phiO  outdoor relative humidity; 100% for design conditions
        
    System:
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions
        0, 1, 2 unknown points (temperature, humidity ratio)

    --o->HC--0->VH--1->TZ--2-->
         |       |     BL  |
         |       |         |
         |       |<----Kw--|-w2
         |<------------Kt--|-t2
    """
    plt.close('all')
    wO = psy.w(tO, phiO)            # hum. out

    # Mass flow rate for design conditions
    # Supplay air mass flow rate
    # QsZ = UA*(tO - tIsp) + mi*c*(tO - tIsp)
    # m = - QsZ/(c*(tS - tIsp)
    # where
    # tO, wO = -1, 3.5e-3           # outdoor
    # tS = 30                       # supply air
    # mi = 2.18                     # infiltration
    QsZ = UA*(-1 - tIsp) + 2.18*c*(-1 - tIsp)
    m = - QsZ/(c*(tS - tIsp))
    print('All Out Air CAV')
    print(f'm = {m: 5.3f} kg/s constant (from design conditions)')
    print('Design conditions tS = {tS: 3.1f} °C,'
          'mi = 2.18 kg/S, tO = -1°C, phi0 = 100%')

    # Model
    x = ModelAllOutAir(m, tS, mi, tO, phiO)

    pd.options.display.float_format = '{:,.1f} °C'.format
    t = pd.Series(x[:6:2], index=['tH', 'tS', 'tI'])
    print('\nTemperature:')
    print(t)

    pd.options.display.float_format = '{:,.2f} g/kg'.format
    w = pd.Series(x[1:6:2], index=['wH', 'wS', 'wI'])
    print('\nHumidity ratio:')
    print(1000*w)

    pd.options.display.float_format = '{:,.0f} kW'.format
    Q = pd.Series(x[6:], index=['QsS', 'QlL', 'QsZ', 'QlZ'])
    print('\nHeat flow:')
    print(Q/1000)

    # Processes on psychrometric chart
    A = np.array([[-1,  1,  0,  0],
                 [0,  -1,  1,  0],
                 [0,   0,  1, -1]])
    t = np.append(tO, x[0:5:2])
    t = np.append(t, 40)

    w1 = np.append(wO, x[1:6:2])
    psy.chartA(t, w1, A)

    return None


def AllOutAirVAV(tSsp=30, mi=2.18, tO=-1, phiO=1):
    """
    All out air
    VAV Variablr Air Volume:
        mass flow rate calculated to have ct. supply temp.

    INPUTS:
        tS    supply air;                30°C for design conditions
        mi    infiltration massflow rate 2.18 kg/s for design conditions
        tO    outdoor temperature;       -1°C for design conditions
        phiO  outdoor relative humidity; 100% for design conditions

    System:
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        BL:     Building
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions
        0, 1, 2 unknown points (temperature, humidity ratio)

    --o->HC--0->VH--F-----1-->TZ--2-->
         |       |  |     |   BL  |
         |       |  |_Kt1_|       |
         |       |                |
         |       |<----Kw---------|-w2
         |<------------Kt---------|-t2
        
        Mass-flow rate (VAV) I-controller:
        start with m = 0
        measure the supply temperature
        while -(tSsp - tS)>0.01, increase m (I-controller)
    """
    plt.close('all')
    wO = psy.w(tO, phiO)            # outdoor mumidity ratio

    # Mass flow rate
    DtS, m = 2, 0                   # initial temp; diff; flow rate
    while DtS > 0.01:
        m = m + 0.01                # mass-flow rate I-controller
        # Model
        x = ModelAllOutAir(m, tSsp, mi, tO, phiO)
        tS = x[2]
        DtS = -(tSsp - tS)
    print('All Out Air VAV')
    print(f'm = {m: 5.3f} kg/s')

    pd.options.display.float_format = '{:,.1f} °C'.format
    t = pd.Series(x[:6:2], index=['tH', 'tS', 'tI'])
    print('\nTemperature:')
    print(t)

    pd.options.display.float_format = '{:,.2f} g/kg'.format
    w = pd.Series(x[1:6:2], index=['wH', 'wS', 'wI'])
    print('\nHumidity ratio:')
    print(1000*w)

    pd.options.display.float_format = '{:,.0f} kW'.format
    Q = pd.Series(x[6:], index=['QsS', 'QlL', 'QsZ', 'QlZ'])
    print('\nHeat flow:')
    print(Q/1000)

    # Processes on psychrometric chart
    A = np.array([[-1,  1,  0,  0],
                 [0,  -1,  1,  0],
                 [0,   0,  1, -1]])
    t = np.append(tO, x[0:5:2])
    t = np.append(t, 40)
    print(f'wO = {wO:6.5f}')
    w1 = np.append(wO, x[1:6:2])
    psy.chartA(t, w1, A)
    return None


# *****************************************
# RECYCLED AIR
# *****************************************


def ModelRecAir(m, tS, mi, tO, phiO, alpha):
    """
    Model:
        Heating and vapor humidification
        Recycled air
        CAV Constant Air Volume:
            mass flow rate calculated for design conditions
            maintained constant in all situations
    INPUTS:
        m     mass flow of supply dry air
        tS    supply air;                30°C for design conditions
        mi    infiltration massflow rate 2.18 kg/s for design conditions
        tO    outdoor temperature;       -1°C for design conditions
        phiO  outdoor relative humidity; 100% for design conditions
        alpha mixing ratio of outdoor air
  
    OUTPUTS:
        x     vector 12 elements:
            t0, w0, t1, w1, t2, w2, t3, w3, QsHC, QlVH, QsTZ, QlTZ

    System:
        MX:     Mixing Box
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        BL:     Buildings
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions
        0, 1, 2 unknown points (temperature, humidity ratio)

    <-3--|<-------------------------|
         |                          |
    -o->MX--0->HC--1->VH--2->TZ--3-->
               |       |     ||  |
               |       |     BL  |
               |       |         |
               |       |<----Kw--|-w3
               |<------------Kt--|-t3
    """
    Kt, Kw = 1e10, 1e10             # controller gain
    wO = psy.w(tO, phiO)            # hum. out

    # Model
    A = np.zeros((12, 12))          # coefficents of unknowns
    b = np.zeros(12)                # vector of inputs

    A[0, 0], A[0, 6], b[0] = m*c, -(1 - alpha)*m*c, alpha*m*c*tO    # MX
    A[1, 1], A[1, 7], b[1] = m*l, -(1 - alpha)*m*l, alpha*m*l*wO

    A[2, 0], A[2, 2], A[2, 8], b[2] = m*c, -m*c, 1, 0               # HC
    A[3, 1], A[3, 3], b[3] = m*l, -m*l, 0

    A[4, 2], A[4, 4], b[4] = m*c, -m*c, 0                           # VH
    A[5, 3], A[5, 5], A[5, 9], b[5] = m*l, -m*l, 1, 0

    A[6, 4], A[6, 6], A[6, 10], b[6] = m*c, -m*c, 1, 0              # TZ
    A[7, 5], A[7, 7], A[7, 11], b[7] = m*l, -m*l, 1, 0

    A[8, 6], A[8, 10], b[8] = (UA + mi*c), 1, (UA + mi*c)*tO        # BL
    A[9, 7], A[9, 11], b[9] = mi*l, 1, mi*l*wO

    A[10, 6], A[10, 8], b[10] = Kt, 1, Kt*tIsp                      # Kt
    A[11, 7], A[11, 9], b[11] = Kw, 1, Kw*wIsp                      # Kw

    # Solution
    x = np.linalg.solve(A, b)
    return x


def RecAirCAV(tS=30, mi=2.18, tO=-1, phiO=1, alpha=0.5):
    """
    CAV Constant Air Volume:
    mass flow rate calculated for design conditions
    maintained constant in all situations
    INPUTS:
    tS    supply air;                30°C for design conditions
    mi    infiltration massflow rate 2.18 kg/s for design conditions
    tO    outdoor temperature;       -1°C for design conditions
    phiO  outdoor relative humidity; 100% for design conditions
    alpha mixing ratio of outdoor air

    System:
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions
        0, 1, 2 unknown points (temperature, humidity ratio)

    <-3--|<-------------------------|
         |                          |
    -o->MX--0->HC--1->VH--2->TZ--3-->
               |       |_____Kw__|_w3
               |_____________Kt__|_t3
    """
    plt.close('all')
    wO = psy.w(tO, phiO)            # hum. out

    # Mass flow rate for design conditions
    # Supplay air mass flow rate
    # QsZ = UA*(tO - tIsp) + mi*c*(tO - tIsp)
    # m = - QsZ/(c*(tS - tIsp)
    # where
    # tO, wO = -1, 3.5e-3           # outdoor
    # tS = 30                       # supply air
    # mi = 2.18                     # infiltration
    QsZ = UA*(-1 - tIsp) + 2.18*c*(-1 - tIsp)
    m = - QsZ/(c*(tS - tIsp))
    print('Rec Air CAV')
    print(f'm = {m: 5.3f} kg/s constant (from design conditions)')
    print(f'Design conditions tS = {tS: 3.1f} °C,'
          'mi = 2.18 kg/S, tO = -1°C, phi0 = 100%')

    # Model
    x = ModelRecAir(m, tS, mi, tO, phiO, alpha)

    # Processes on psychrometric chart
    A = np.array([[-1,  1,  0,  0, -1],
                 [0,  -1,  1,  0,   0],
                 [0,   0,  -1, 1,   0],
                 [0,   0,  0, -1,   1]])
    t = np.append(tO, x[0:8:2])
    t = np.append(t, 40)

    print(f'wO = {wO:6.5f}')
    w1 = np.append(wO, x[1:8:2])
    psy.chartA(t, w1, A)

    pd.options.display.float_format = '{:,.1f} °C'.format
    t = pd.Series(x[:8:2], index=['tM', 'tH', 'tS', 'tI'])
    print('\nTemperature:')
    print(t)

    pd.options.display.float_format = '{:,.2f} g/kg'.format
    w = pd.Series(x[1:8:2], index=['wM', 'wH', 'wS', 'wI'])
    print('\nHumidity ratio:')
    print(1000*w)

    pd.options.display.float_format = '{:,.0f} kW'.format
    Q = pd.Series(x[8:], index=['QsS', 'QlL', 'QsZ', 'QlZ'])
    print('\nHeat flow:')
    print(Q/1000)

    return None


def RecAirVAV(tSsp=30, mi=2.18, tO=-1, phiO=1, alpha=0.5):
    """
    CAV Variable Air Volume:
    mass flow rate calculated s.t.
    he supply temp. is maintained constant in all situations
    INPUTS:
    tS    supply air;                30°C for design conditions
    mi    infiltration massflow rate 2.18 kg/s for design conditions
    tO    outdoor temperature;       -1°C for design conditions
    phiO  outdoor relative humidity; 100% for design conditions
    alpha ratio of out air: alpha = 1 -> All out air;
                            alpha = 0 -> All recirculated air

    System (CAV & m introduced by the Fan is cotrolled by tS )
        HC:     Heating Coil
        VH:     Vapor Humidifier
        TZ:     Thermal Zone
        F:      Supply air fan
        Kw:     Controller - humidity
        Kt:     Controller - temperature
        o:      outdoor conditions
        0, 1, 2, 3 unknown points (temperature, humidity ratio)

    <----|<--------------------------------|
         |                                 |
    -o->MX--0->HC--1->VH--F-----2-->TZ--3-->
               |       |  |     |       |
               |       |  |_Kt2_|_t2    |
               |       |                |
               |       |_____Kw_________|_w3
               |_____________Kt_________|_t3

    Mass-flow rate (VAV) I-controller:
        start with m = 0
        measure the supply temperature
        while -(tSsp - tS)>0.01, increase m (I-controller)
    """
    plt.close('all')
    wO = psy.w(tO, phiO)            # hum. out

    # Mass flow rate
    DtS, m = 2, 0                   # initial temp; diff; flow rate
    while DtS > 0.01:
        m = m + 0.01

        # Model
        x = ModelRecAir(m, tSsp, mi, tO, phiO, alpha)
        tS = x[4]
        DtS = -(tSsp - tS)

    print('Rec Air VAV')
    print(f'm = {m: 5.3f} kg/s')

    pd.options.display.float_format = '{:,.1f} °C'.format
    t = pd.Series(x[:8:2], index=['tM', 'tH', 'tS', 'tI'])
    print('\nTemperature:')
    print(t)

    pd.options.display.float_format = '{:,.2f} g/kg'.format
    w = pd.Series(x[1:8:2], index=['wM', 'wH', 'wS', 'wI'])
    print('\nHumidity ratio:')
    print(1000*w)

    pd.options.display.float_format = '{:,.0f} kW'.format
    Q = pd.Series(x[8:], index=['QsS', 'QlL', 'QsZ', 'QlZ'])
    print('\nHeat flow:')
    print(Q/1000)

    # Processes on psychrometric chart
    A = np.array([[-1,  1,  0,  0, -1],
                 [0,  -1,  1,  0,   0],
                 [0,   0,  -1, 1,   0],
                 [0,   0,  0, -1,   1]])
    t = np.append(tO, x[0:8:2])
    t = np.append(t, 40)
    print(f'wO = {wO:6.5f}')
    w1 = np.append(wO, x[1:8:2])
    psy.chartA(t, w1, A)
    return None


# AllOutAirCAV()
# AllOutAirVAV()
# RecAirCAV()
# RecAirVAV()
