import config
import pandas as pd
import pandas_ta_classic as ta


class TechnicalIndicators:
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw OHLCV data into a state vector for the AI.
        Separates indicators into logical groups:
        - Momentum
        - Overlap
        - Performance
        - Statistics
        - Trend
        - Volatility
        - Volume
        """
        self.__add_momentum_indicators(df)
        self.__add_overlap_indicators(df)
        self.__add_performance_indicators(df)
        self.__add_statistics_indicators(df)
        self.__add_trend_indicators(df)
        self.__add_volatility_indicators(df)
        self.__add_volume_indicators(df)

        df.dropna(inplace=True)
        return df

    # --- Momentum Indicators ---

    def __add_momentum_indicators(self, df: pd.DataFrame) -> None:
        """Adds Momentum Indicators."""
        self.__calculate_ao(df)
        self.__calculate_macd(df)
        self.__calculate_rsi(df)
        self.__calculate_stochrsi(df)
        self.__calculate_squeeze(df)

    def __calculate_ao(self, df: pd.DataFrame) -> None:
        """Calculates Awesome Oscillator."""
        ao = ta.ao(df["high"], df["low"], fast=config.AO_FAST, slow=config.AO_SLOW)
        if ao is not None and not ao.empty:
            df["ao"] = ao

    def __calculate_macd(self, df: pd.DataFrame) -> None:
        """Calculates MACD and MACD Histogram."""
        macd = ta.macd(df["close"], fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL)
        if macd is not None and not macd.empty:
            df["macd"] = macd[f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
            df["macd_signal"] = macd[f"MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
            df["macd_hist"] = macd[f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]

    def __calculate_rsi(self, df: pd.DataFrame) -> None:
        """Calculates RSI."""
        rsi = ta.rsi(df["close"], length=config.RSI_LENGTH)
        if rsi is not None and not rsi.empty:
            df["rsi"] = rsi

    def __calculate_stochrsi(self, df: pd.DataFrame) -> None:
        """Calculates Stochastic RSI."""
        stochrsi = ta.stochrsi(df["close"], length=config.STOCHRSI_LENGTH, rsi_length=config.STOCHRSI_RSI_LENGTH, k=config.STOCHRSI_K, d=config.STOCHRSI_D)
        if stochrsi is not None and not stochrsi.empty:
            df["stochrsi_k"] = stochrsi[f"STOCHRSIk_{config.STOCHRSI_LENGTH}_{config.STOCHRSI_RSI_LENGTH}_{config.STOCHRSI_K}_{config.STOCHRSI_D}"]
            df["stochrsi_d"] = stochrsi[f"STOCHRSId_{config.STOCHRSI_LENGTH}_{config.STOCHRSI_RSI_LENGTH}_{config.STOCHRSI_K}_{config.STOCHRSI_D}"]

    def __calculate_squeeze(self, df: pd.DataFrame) -> None:
        """Calculates Squeeze."""
        squeeze = ta.squeeze(df["high"], df["low"], df["close"], bb_length=config.BBANDS, bb_std=config.BBANDS_STD, kc_length=config.SQUEEZE_KC_LENGTH, kc_scalar=config.SQUEEZE_KC_SCALAR, mom_length=config.SQUEEZE_MOM_LENGTH, mom_smooth=config.SQUEEZE_MOM_SMOOTH, mamode=config.SQUEEZE_MA_MODE)
        if squeeze is not None and not squeeze.empty:
            df["sqz"] = squeeze[f"SQZ_{config.BBANDS}_{config.BBANDS_STD}_{config.SQUEEZE_KC_LENGTH}_{config.SQUEEZE_KC_SCALAR}"]
            df["sqz_on"] = squeeze["SQZ_ON"]
            df["sqz_off"] = squeeze["SQZ_OFF"]
            df["no_sqz"] = squeeze["NO_SQZ"]

    # --- Overlap Indicators ---

    def __add_overlap_indicators(self, df: pd.DataFrame) -> None:
        """Adds Overlap Indicators."""
        self.__calculate_alma(df)
        self.__calculate_ema(df)
        self.__calculate_hma(df)
        self.__calculate_super_trend(df)
        self.__calculate_vwap(df)

    def __calculate_alma(self, df: pd.DataFrame) -> None:
        """Calculates Arnaud Legoux Moving Average."""
        alma = ta.alma(df["close"], length=config.ALMA_LENGTH, sigma=config.ALMA_SIGMA, distribution_offset=config.ALMA_DISTRIBUTION_OFFSET)
        if alma is not None and not alma.empty:
            df["alma"] = alma

    def __calculate_ema(self, df: pd.DataFrame) -> None:
        """Calculates Exponential Moving Average."""
        ema = ta.ema(df["close"], length=config.EMA_LENGTH)
        if ema is not None and not ema.empty:
            df["ema"] = ema

    def __calculate_hma(self, df: pd.DataFrame) -> None:
        """Calculates Hull Moving Average."""
        hma = ta.hma(df["close"], length=config.HMA_LENGTH)
        if hma is not None and not hma.empty:
            df["hma"] = hma

    def __calculate_super_trend(self, df: pd.DataFrame) -> None:
        """Calculates Super Trend."""
        super_trend = ta.supertrend(df["high"], df["low"], df["close"], length=config.SUPER_TREND_LENGTH, multiplier=config.SUPER_TREND_MULTIPLIER)
        if super_trend is not None and not super_trend.empty:
            df["super_trend"] = super_trend[f"SUPERT_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
            df["super_trend_direction"] = super_trend[f"SUPERTd_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
            df["super_trend_long"] = super_trend[f"SUPERTl_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]
            df["super_trend_short"] = super_trend[f"SUPERTs_{config.SUPER_TREND_LENGTH}_{config.SUPER_TREND_MULTIPLIER}"]

    def __calculate_vwap(self, df: pd.DataFrame) -> None:
        """Calculates Volume Weighted Average Price."""
        vwap = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        if vwap is not None and not vwap.empty:
            df["vwap"] = vwap

    # --- Performance Indicators ---

    def __add_performance_indicators(self, df: pd.DataFrame) -> None:
        """Adds Performance Indicators."""
        self.__calculate_draw_down(df)
        self.__calculate_log_return(df)
        self.__calculate_percent_return(df)

    def __calculate_draw_down(self, df: pd.DataFrame) -> None:
        """Calculates Draw Down."""
        draw_down = ta.drawdown(df["close"])
        if draw_down is not None and not draw_down.empty:
            df["draw_down"] = draw_down["DD"]
            df["draw_down_pct"] = draw_down["DD_PCT"]
            df["draw_down_log"] = draw_down["DD_LOG"]

    def __calculate_log_return(self, df: pd.DataFrame) -> None:
        """Calculates Log Return."""
        log_return = ta.log_return(df["close"], length=config.LOG_RETURN_LENGTH)
        if log_return is not None and not log_return.empty:
            df["log_return"] = log_return

    def __calculate_percent_return(self, df: pd.DataFrame) -> None:
        """Calculates Percent Return."""
        pct_return = ta.percent_return(df["close"], length=config.PERCENT_RETURN_LENGTH)
        if pct_return is not None and not pct_return.empty:
            df["pct_return"] = pct_return

    # --- Statistics Indicators ---

    def __add_statistics_indicators(self, df: pd.DataFrame) -> None:
        """Adds Statistics Indicators."""
        self.__calculate_entropy(df)
        self.__calculate_mad(df)
        self.__calculate_stdev(df)
        self.__calculate_variance(df)
        self.__calculate_zscore(df)

    def __calculate_entropy(self, df: pd.DataFrame) -> None:
        """Calculates Entropy."""
        entropy = ta.entropy(df["close"], length=config.ENTROPY_LENGTH, base=config.ENTROPY_BASE)
        if entropy is not None and not entropy.empty:
            df["entropy"] = entropy

    def __calculate_mad(self, df: pd.DataFrame) -> None:
        """Calculates Mean Absolute Deviation."""
        mad = ta.mad(df["close"], length=config.MAD_LENGTH)
        if mad is not None and not mad.empty:
            df["mad"] = mad

    def __calculate_stdev(self, df: pd.DataFrame) -> None:
        """Calculates Standard Deviation."""
        stdev = ta.stdev(df["close"], length=config.STD_DEV_LENGTH, ddof=config.STD_DEV_DDOF)
        if stdev is not None and not stdev.empty:
            df["stdev"] = stdev

    def __calculate_variance(self, df: pd.DataFrame) -> None:
        """Calculates Variance."""
        variance = ta.variance(df["close"], length=config.VARIANCE_LENGTH, ddof=config.VARIANCE_DDOF)
        if variance is not None and not variance.empty:
            df["variance"] = variance

    def __calculate_zscore(self, df: pd.DataFrame) -> None:
        """Calculates Z-Score."""
        zscore = ta.zscore(df["close"], length=config.ZSCORE_LENGTH, std=config.ZSCORE_STD)
        if zscore is not None and not zscore.empty:
            df["zscore"] = zscore

    # --- Trend Indicators ---

    def __add_trend_indicators(self, df: pd.DataFrame) -> None:
        """Adds Trend Indicators."""
        self.__calculate_adx(df)
        self.__calculate_aroon(df)
        self.__calculate_chop(df)
        self.__calculate_psar(df)
        self.__calculate_vortex(df)

    def __calculate_adx(self, df: pd.DataFrame) -> None:
        """Calculates Average Directional Index."""
        adx = ta.adx(df["high"], df["low"], df["close"], length=config.ADX_LENGTH)
        if adx is not None and not adx.empty:
            df["adx"] = adx[f"ADX_{config.ADX_LENGTH}"]
            df["dmp"] = adx[f"DMP_{config.ADX_LENGTH}"]
            df["dmn"] = adx[f"DMN_{config.ADX_LENGTH}"]

    def __calculate_aroon(self, df: pd.DataFrame) -> None:
        """Calculates Aroon & Aroon Osciallator."""
        aroon = ta.aroon(df["high"], df["low"], length=config.AROON_LENGTH, scalar=config.AROON_SCALAR)
        if aroon is not None and not aroon.empty:
            df["aroon_up"] = aroon[f"AROONU_{config.AROON_LENGTH}"]
            df["aroon_down"] = aroon[f"AROOND_{config.AROON_LENGTH}"]
            df["aroon_osc"] = aroon[f"AROONOSC_{config.AROON_LENGTH}"]

    def __calculate_chop(self, df: pd.DataFrame) -> None:
        """Calculates Choppiness Index."""
        chop = ta.chop(df["high"], df["low"], df["close"], length=config.CHOP_LENGTH, atr_length=config.CHOP_ATR_LENGTH, ln=config.CHOP_LN, scalar=config.CHOP_SCALAR)
        if chop is not None and not chop.empty:
            df["chop"] = chop

    def __calculate_psar(self, df: pd.DataFrame) -> None:
        """Calculates Parabolic Stop and Reverse."""
        psar = ta.psar(df["high"], df["low"], df["close"], af0=config.PSAR_INIT_ACC, af=config.PSAR_ACC, max_af=config.PSAR_MAX_ACC)
        if psar is not None and not psar.empty:
            df["psar_l"] = psar[f"PSARl_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
            df["psar_s"] = psar[f"PSARs_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
            df["psar_af"] = psar[f"PSARaf_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]
            df["psar_r"] = psar[f"PSARr_{config.PSAR_INIT_ACC}_{config.PSAR_MAX_ACC}"]

    def __calculate_vortex(self, df: pd.DataFrame) -> None:
        """Calculates Vortex."""
        vortex = ta.vortex(df["high"], df["low"], df["close"], length=config.VORTEX_LENGTH)
        if vortex is not None and not vortex.empty:
            df["vortex_p"] = vortex[f"VTXP_{config.VORTEX_LENGTH}"]
            df["vortex_m"] = vortex[f"VTXM_{config.VORTEX_LENGTH}"]

    # --- Volatility Indicators ---

    def __add_volatility_indicators(self, df: pd.DataFrame) -> None:
        """Adds Volatility Indicators."""
        self.__calculate_atr(df)
        self.__calculate_bbands(df)
        self.__calculate_donchian(df)
        self.__calculate_kc(df)
        self.__calculate_ui(df)

    def __calculate_atr(self, df: pd.DataFrame) -> None:
        """Calculates Average True Range."""
        atr = ta.atr(df["high"], df["low"], df["close"], length=config.ATR)
        if atr is not None and not atr.empty:
            df["atr"] = atr

    def __calculate_bbands(self, df: pd.DataFrame) -> None:
        """Calculates Bollinger Bands Width."""
        bb = ta.bbands(df["close"], length=config.BBANDS, std=config.BBANDS_STD)
        if bb is not None and not bb.empty:
            df["bb_lower"] = bb[f"BBL_{config.BBANDS}_{config.BBANDS_STD}"]
            df["bb_mid"] = bb[f"BBM_{config.BBANDS}_{config.BBANDS_STD}"]
            df["bb_upper"] = bb[f"BBU_{config.BBANDS}_{config.BBANDS_STD}"]
            df["bb_width"] = bb[f"BBB_{config.BBANDS}_{config.BBANDS_STD}"]
            df["bb_pct"] = bb[f"BBP_{config.BBANDS}_{config.BBANDS_STD}"]

    def __calculate_donchian(self, df: pd.DataFrame) -> None:
        """Calculates Donchian Channel."""
        donchian = ta.donchian(df["high"], df["low"], lower_length=config.DONCHIAN_LOWER_LENGTH, upper_length=config.DONCHIAN_UPPER_LENGTH)
        if donchian is not None and not donchian.empty:
            df["donchian_l"] = donchian[f"DCL_{config.DONCHIAN_LOWER_LENGTH}_{config.DONCHIAN_UPPER_LENGTH}"]
            df["donchian_m"] = donchian[f"DCM_{config.DONCHIAN_LOWER_LENGTH}_{config.DONCHIAN_UPPER_LENGTH}"]
            df["donchian_u"] = donchian[f"DCU_{config.DONCHIAN_LOWER_LENGTH}_{config.DONCHIAN_UPPER_LENGTH}"]

    def __calculate_kc(self, df: pd.DataFrame) -> None:
        """Calculate Keltner Channel."""
        kc = ta.kc(df["high"], df["low"], df["close"], length=config.KC_LENGTH, scalar=config.KC_SCALAR)
        if kc is not None and not kc.empty:
            df["kc_l"] = kc[f"KCLe_{config.KC_LENGTH}_{config.KC_SCALAR}"]
            df["kc_b"] = kc[f"KCBe_{config.KC_LENGTH}_{config.KC_SCALAR}"]
            df["kc_u"] = kc[f"KCUe_{config.KC_LENGTH}_{config.KC_SCALAR}"]

    def __calculate_ui(self, df: pd.DataFrame) -> None:
        """Calculate Ulcer Index."""
        ui = ta.ui(df["close"], length=config.UI_LENGTH, scalar=config.UI_SCALAR)
        if ui is not None and not ui.empty:
            df["ui"] = ui

    # --- Volume Indicators ---

    def __add_volume_indicators(self, df: pd.DataFrame) -> None:
        """Adds Volume Indicators."""
        self.__calculate_cmf(df)
        self.__calculate_efi(df)
        self.__calculate_mfi(df)
        self.__calculate_obv(df)
        self.__calculate_vp(df)

    def __calculate_cmf(self, df: pd.DataFrame) -> None:
        """Calculates Chaikin Money Flow."""
        cmf = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=config.CMF_LENGTH)
        if cmf is not None and not cmf.empty:
            df["cmf"] = cmf

    def __calculate_efi(self, df: pd.DataFrame) -> None:
        """Calculates Elder Force Index."""
        efi = ta.efi(df["close"], df["volume"], length=config.EFI_LENGTH)
        if efi is not None and not efi.empty:
            df["efi"] = efi

    def __calculate_mfi(self, df: pd.DataFrame) -> None:
        """Calculates Money Flow Index."""
        mfi = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=config.MFI_LENGTH)
        if mfi is not None and not mfi.empty:
            df["mfi"] = mfi

    def __calculate_obv(self, df: pd.DataFrame) -> None:
        """Calculates On-Balance Volume."""
        obv = ta.obv(df["close"], df["volume"])
        if obv is not None and not obv.empty:
            df["obv"] = obv

    def __calculate_vp(self, df: pd.DataFrame) -> None:
        """Calculates Volume Profile."""
        vp = ta.vp(df["close"], df["volume"], width=config.VP_LENGTH)
        if vp is not None and not vp.empty:
            df["vp_low"] = vp["low_close"]
            df["vp_mean"] = vp["mean_close"]
            df["vp_high"] = vp["high_close"]
            df["vp_pos"] = vp["pos_volume"]
            df["vp_neg"] = vp["neg_volume"]
            df["vp_total"] = vp["total_volume"]
