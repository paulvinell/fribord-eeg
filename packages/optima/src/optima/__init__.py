"""Neurobit Optima driver for macOS.

A from-scratch implementation of the Optima's wire protocol over its FTDI serial link,
replacing the Windows-only vendor driver. See AGENTS.md for how the hardware actually behaves.

Layout: `protocol` is framing and opcodes, `tables` is the parameter data recovered from the
vendor driver, `config` is the configuration blob, `stream` is sample decoding, `device` is the
serial session, `cli` is the command line.
"""

from .protocol import (
    BAUD, FLOW_RTSCTS, CLASS_CMD, CLASS_REP, CLASS_NOT, CMD, CMD_NAME,
    NOT_EVENT, NOT_DATA, NOT_EXT_DATA, NOT_NAME, STATUS, REP_MINLEN, MAX_PAYLOAD,
    NOTIF_ENABLE, NOTIF_CLEAR, EVENT_BATTERY, EVENT_LINK_ERROR, EVENT_STOPPED,
    EVENT_HARDWARE, EVENT_NAME, SIGNAL_STATE, DEV_STATE, IDLE_POWEROFF_SECONDS,
    ProtocolError, DeviceError, Message, encode_varint, encode,
)
from .tables import (
    FUNC_VOLTAGE, FUNC_RESISTANCE, FUNC_CONDUCTANCE, FUNC_TEMPERATURE, FUNC_NIR_HEG,
    FUNC_PIR_HEG, FUNC_CURRENT, FUNC_BUTTON, FUNC_NAME, PROFILE, FREQ_CHAR, COMMON_REF,
    EEG_CAP_NONE, EEG_CAP_CODES, EEG_CAP_SITES, ADC_FULL_SCALE, MAX_SAMPLE_STREAM,
    RANGE_BY_FUNCTION, RANGE_MIN_BY_FUNCTION, RATES_BY_FUNCTION, DEFAULT_RATE_EXPONENT,
    LOWER_LIMIT_BY_FUNCTION, FREQ_CHAR_BY_FUNCTION, TEST_RANGE_BY_FUNCTION,
    COMMON_REF_BY_FUNCTION, NOTCH_DEFAULT, SUM_DISCONNECTED_DEFAULT, TEST_RATE_EXPONENT,
    TEST_RATE_HZ, TEST_UNIT, RANGE_TEST_KOHM, TEST_RANGE_DEFAULT,
    test_scale, cap_code, rate_hz, rate_exponent,
)
from .config import (
    CFG_BASE, CFG_HEADER, CFG_CHANNEL_STRIDE, CFG_MAX_CHANNELS, CFG_RECORDS,
    LINK_BLUETOOTH, LINK_USB, HEADER_FIELDS, HEADER_DATA_ENABLE, CHANNEL_FIELDS,
    CH_RATE_EXPONENT, REFERENCE_MODES, get_field, set_field, channel_range, channel_scale, valid_span,
    lower_limit_hz, to_units, validate_config, build_eeg_config, contact_check_config, describe_config,
    config_size,
)
from .stream import ChannelPlan, plan_channels, DataFrame, decode_data
from .device import Optima, find_port
from .cli import probe, capture, impedance, main

__all__ = [n for n in dir() if not n.startswith("_")]
