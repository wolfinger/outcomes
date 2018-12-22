# -*- coding: utf-8 -*-
"""Test driver for the outcomes package."""

import sys, os

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path)

import pandas as pd
import outcomes.outcomes as outcomes


def test_chg_analysis(src_file=None, src_sheet='Sheet1', src_col_dt='date',
         src_col_topic='topic', src_col_rfr='rfr', src_col_bmk='benchmark',
         tgt_file=None, tgt_sheet='Sheet1'):
    """Test the change_analysis funciton."""

    # read source data
    path = os.path.dirname(os.path.abspath(__file__)) + '/'
    src_file = 'src_returns.xlsx'
    tgt_file = 'out_returns.xlsx'
    xlsx = pd.ExcelFile(path + src_file)
    src_df = pd.read_excel(xlsx, src_sheet)
    src_df.index = src_df[src_col_dt]
    
    # test measures
    measures_labels = outcomes.Measures()
    measures = [measures_labels.level, measures_labels.level_ln, 
                measures_labels.chg_rel, measures_labels.chg_ln, 
                measures_labels.vol_ln]
    measures_exclude_flag = False
    periods = [outcomes.TP_1M, outcomes.TP_3M, outcomes.TP_6M, outcomes.TP_1Y, 
               outcomes.TP_CUM]
    df = outcomes.change_analysis(src_df, src_col_topic, src_col_rfr, 
                                  src_col_bmk, measures=measures, 
                                  m_lbls=measures_labels, 
                                  measures_exclude_flag=measures_exclude_flag, 
                                  periods=periods)
    xlsx_writer = pd.ExcelWriter(path + tgt_file)
    df.reorder_levels(('measure', 'period', 'srs_type'), 1).sort_index(axis=1,
                      level=('measure', 'period', 'srs_type')).to_excel(
                          xlsx_writer, tgt_sheet)
    xlsx_writer.save()


if __name__ == '__main__':
    test_chg_analysis()