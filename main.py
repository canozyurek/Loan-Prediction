from sqlite3 import Date
from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
from pydantic import BaseModel
from pycaret.classification import *
import pickle
import datetime
from category_encoders import *
#from helpers.transformers import DateTransformer
import joblib

app = FastAPI(title="Loan Status Prediction App")

@app.get("/")
@app.get("/home")
def read_home():

    return {"message": "System is healthy"}


class ApplicationData(BaseModel):
    date: datetime.date
    loan_amnt: float
    dti: float
    emp_length: str


@app.post("/predict-application-status")
def predict_stroke(data: ApplicationData):
    data_frame = pd.DataFrame(
        data=np.array([data.date, data.loan_amnt, data.dti,
                      data.emp_length]).reshape(1, 4),
        columns=[
            "date",
            "loan_amnt",
            "dti",
            "emp_length"]
    )
    app_model = load_model('models/app_status_final')
    prediction = predict_model(app_model, data_frame)[['Label', 'Score']]
    return f"{prediction['Label'].values[0]}: {prediction['Score'].values[0]}"


class GradeData(BaseModel):
    issue_d: datetime.date
    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: float
    term: str
    installment: float
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    loan_status: str
    pymnt_plan: str
    purpose: str
    title: str
    dti: float
    delinq_2yrs: float
    earliest_cr_line: datetime.date
    fico_range_low: float
    fico_range_high: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    initial_list_status: str
    out_prncp: float
    out_prncp_inv: float
    total_pymnt: float
    total_pymnt_inv: float
    total_rec_prncp: float
    total_rec_int: float
    total_rec_late_fee: float
    recoveries: float
    collection_recovery_fee: float
    last_pymnt_d: datetime.date
    last_pymnt_amnt: float
    next_pymnt_d: datetime.date
    last_credit_pull_d: datetime.date
    last_fico_range_high: float
    last_fico_range_low: float
    collections_12_mths_ex_med: float
    application_type: str
    annual_inc_joint: float
    dti_joint: float
    verification_status_joint: str
    acc_now_delinq: float
    tot_coll_amt: float
    tot_cur_bal: float
    open_acc_6m: float
    open_act_il: float
    open_il_12m: float
    open_il_24m: float
    mths_since_rcnt_il: float
    total_bal_il: float
    il_util: float
    open_rv_12m: float
    open_rv_24m: float
    max_bal_bc: float
    all_util: float
    total_rev_hi_lim: float
    inq_fi: float
    total_cu_tl: float
    inq_last_12m: float
    acc_open_past_24mths: float
    avg_cur_bal: float
    bc_open_to_buy: float
    bc_util: float
    chargeoff_within_12_mths: float
    delinq_amnt: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mo_sin_rcnt_tl: float
    mort_acc: float
    mths_since_recent_bc: float
    mths_since_recent_inq: float
    num_accts_ever_120_pd: float
    num_actv_bc_tl: float
    num_actv_rev_tl: float
    num_bc_sats: float
    num_bc_tl: float
    num_il_tl: float
    num_op_rev_tl: float
    num_rev_accts: float
    num_rev_tl_bal_gt_0: float
    num_sats: float
    num_tl_120dpd_2m: float
    num_tl_30dpd: float
    num_tl_90g_dpd_24m: float
    num_tl_op_past_12m: float
    pct_tl_nvr_dlq: float
    percent_bc_gt_75: float
    pub_rec_bankruptcies: float
    tax_liens: float
    tot_hi_cred_lim: float
    total_bal_ex_mort: float
    total_bc_limit: float
    total_il_high_credit_limit: float
    revol_bal_joint: bool
    hardship_flag: str
    disbursement_method: str
    debt_settlement_flag: str
    sec_app_flag: bool


@app.post("/predict-grade")
def predict_grade(data: GradeData):
    data_frame = pd.DataFrame(
        data=np.array([data.issue_d, data.loan_amnt, data.funded_amnt, data.funded_amnt_inv, data.term, data.installment, data.emp_length, data.home_ownership, data.annual_inc, data.verification_status, data.loan_status, data.pymnt_plan, data.purpose, data.title, data.dti, data.delinq_2yrs, data.earliest_cr_line, data.fico_range_low, data.fico_range_high, data.inq_last_6mths, data.open_acc, data.pub_rec, data.revol_bal, data.revol_util, data.total_acc, data.initial_list_status, data.out_prncp, data.out_prncp_inv, data.total_pymnt, data.total_pymnt_inv, data.total_rec_prncp, data.total_rec_int, data.total_rec_late_fee, data.recoveries, data.collection_recovery_fee, data.last_pymnt_d, data.last_pymnt_amnt, data.next_pymnt_d, data.last_credit_pull_d, data.last_fico_range_high, data.last_fico_range_low, data.collections_12_mths_ex_med, data.application_type, data.annual_inc_joint, data.dti_joint, data.verification_status_joint, data.acc_now_delinq, data.tot_coll_amt, data.tot_cur_bal, data.open_acc_6m, data.open_act_il, data.open_il_12m, data.open_il_24m, data.mths_since_rcnt_il, data.total_bal_il, data.il_util, data.open_rv_12m, data.open_rv_24m, data.max_bal_bc, data.all_util, data.total_rev_hi_lim, data.inq_fi, data.total_cu_tl, data.inq_last_12m, data.acc_open_past_24mths, data.avg_cur_bal, data.bc_open_to_buy, data.bc_util, data.chargeoff_within_12_mths, data.delinq_amnt, data.mo_sin_old_il_acct, data.mo_sin_old_rev_tl_op, data.mo_sin_rcnt_rev_tl_op, data.mo_sin_rcnt_tl, data.mort_acc, data.mths_since_recent_bc, data.mths_since_recent_inq, data.num_accts_ever_120_pd, data.num_actv_bc_tl, data.num_actv_rev_tl, data.num_bc_sats, data.num_bc_tl, data.num_il_tl, data.num_op_rev_tl, data.num_rev_accts, data.num_rev_tl_bal_gt_0, data.num_sats, data.num_tl_120dpd_2m, data.num_tl_30dpd, data.num_tl_90g_dpd_24m, data.num_tl_op_past_12m, data.pct_tl_nvr_dlq, data.percent_bc_gt_75, data.pub_rec_bankruptcies, data.tax_liens, data.tot_hi_cred_lim, data.total_bal_ex_mort, data.total_bc_limit, data.total_il_high_credit_limit, data.revol_bal_joint, data.hardship_flag, data.disbursement_method, data.debt_settlement_flag, data.sec_app_flag]).reshape(1, 104),
        columns=['issue_d', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'installment', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'loan_status', 'pymnt_plan', 'purpose', 'title', 'dti', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'collections_12_mths_ex_med', 'application_type', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'revol_bal_joint', 'hardship_flag', 'disbursement_method', 'debt_settlement_flag', 'sec_app_flag'])

    grade_model = load_model('models/grade_final')
    prediction = predict_model(grade_model, data_frame)[['Label','Score']]
    return f"{prediction['Label'].values[0]}: {prediction['Score'].values[0]}"


class SubgradeData(BaseModel):
    issue_d: datetime.date
    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: float
    term: str
    installment: float
    grade: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    loan_status: str
    pymnt_plan: str
    purpose: str
    title: str
    dti: float
    delinq_2yrs: float
    earliest_cr_line: datetime.date
    fico_range_low: float
    fico_range_high: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    initial_list_status: str
    out_prncp: float
    out_prncp_inv: float
    total_pymnt: float
    total_pymnt_inv: float
    total_rec_prncp: float
    total_rec_int: float
    total_rec_late_fee: float
    recoveries: float
    collection_recovery_fee: float
    last_pymnt_d: datetime.date
    last_pymnt_amnt: float
    next_pymnt_d: datetime.date
    last_credit_pull_d: datetime.date
    last_fico_range_high: float
    last_fico_range_low: float
    collections_12_mths_ex_med: float
    application_type: str
    annual_inc_joint: float
    dti_joint: float
    verification_status_joint: str
    acc_now_delinq: float
    tot_coll_amt: float
    tot_cur_bal: float
    open_acc_6m: float
    open_act_il: float
    open_il_12m: float
    open_il_24m: float
    mths_since_rcnt_il: float
    total_bal_il: float
    il_util: float
    open_rv_12m: float
    open_rv_24m: float
    max_bal_bc: float
    all_util: float
    total_rev_hi_lim: float
    inq_fi: float
    total_cu_tl: float
    inq_last_12m: float
    acc_open_past_24mths: float
    avg_cur_bal: float
    bc_open_to_buy: float
    bc_util: float
    chargeoff_within_12_mths: float
    delinq_amnt: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mo_sin_rcnt_tl: float
    mort_acc: float
    mths_since_recent_bc: float
    mths_since_recent_inq: float
    num_accts_ever_120_pd: float
    num_actv_bc_tl: float
    num_actv_rev_tl: float
    num_bc_sats: float
    num_bc_tl: float
    num_il_tl: float
    num_op_rev_tl: float
    num_rev_accts: float
    num_rev_tl_bal_gt_0: float
    num_sats: float
    num_tl_120dpd_2m: float
    num_tl_30dpd: float
    num_tl_90g_dpd_24m: float
    num_tl_op_past_12m: float
    pct_tl_nvr_dlq: float
    percent_bc_gt_75: float
    pub_rec_bankruptcies: float
    tax_liens: float
    tot_hi_cred_lim: float
    total_bal_ex_mort: float
    total_bc_limit: float
    total_il_high_credit_limit: float
    revol_bal_joint: bool
    hardship_flag: str
    disbursement_method: str
    debt_settlement_flag: str
    sec_app_flag: bool


@app.post("/predict-subgrade")
def predict_subgrade(data: SubgradeData):
    data_frame = pd.DataFrame(
        data=np.array([data.issue_d, data.loan_amnt, data.funded_amnt, data.funded_amnt_inv, data.term, data.installment, data.grade, data.emp_length, data.home_ownership, data.annual_inc, data.verification_status, data.loan_status, data.pymnt_plan, data.purpose, data.title, data.dti, data.delinq_2yrs, data.earliest_cr_line, data.fico_range_low, data.fico_range_high, data.inq_last_6mths, data.open_acc, data.pub_rec, data.revol_bal, data.revol_util, data.total_acc, data.initial_list_status, data.out_prncp, data.out_prncp_inv, data.total_pymnt, data.total_pymnt_inv, data.total_rec_prncp, data.total_rec_int, data.total_rec_late_fee, data.recoveries, data.collection_recovery_fee, data.last_pymnt_d, data.last_pymnt_amnt, data.next_pymnt_d, data.last_credit_pull_d, data.last_fico_range_high, data.last_fico_range_low, data.collections_12_mths_ex_med, data.application_type, data.annual_inc_joint, data.dti_joint, data.verification_status_joint, data.acc_now_delinq, data.tot_coll_amt, data.tot_cur_bal, data.open_acc_6m, data.open_act_il, data.open_il_12m, data.open_il_24m, data.mths_since_rcnt_il, data.total_bal_il, data.il_util, data.open_rv_12m, data.open_rv_24m, data.max_bal_bc, data.all_util, data.total_rev_hi_lim, data.inq_fi, data.total_cu_tl, data.inq_last_12m, data.acc_open_past_24mths, data.avg_cur_bal, data.bc_open_to_buy, data.bc_util, data.chargeoff_within_12_mths, data.delinq_amnt, data.mo_sin_old_il_acct, data.mo_sin_old_rev_tl_op, data.mo_sin_rcnt_rev_tl_op, data.mo_sin_rcnt_tl, data.mort_acc, data.mths_since_recent_bc, data.mths_since_recent_inq, data.num_accts_ever_120_pd, data.num_actv_bc_tl, data.num_actv_rev_tl, data.num_bc_sats, data.num_bc_tl, data.num_il_tl, data.num_op_rev_tl, data.num_rev_accts, data.num_rev_tl_bal_gt_0, data.num_sats, data.num_tl_120dpd_2m, data.num_tl_30dpd, data.num_tl_90g_dpd_24m, data.num_tl_op_past_12m, data.pct_tl_nvr_dlq, data.percent_bc_gt_75, data.pub_rec_bankruptcies, data.tax_liens, data.tot_hi_cred_lim, data.total_bal_ex_mort, data.total_bc_limit, data.total_il_high_credit_limit, data.revol_bal_joint, data.hardship_flag, data.disbursement_method, data.debt_settlement_flag, data.sec_app_flag]).reshape(1, 105),
        columns=['issue_d', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'installment', 'grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'loan_status', 'pymnt_plan', 'purpose', 'title', 'dti', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low', 'collections_12_mths_ex_med', 'application_type', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'revol_bal_joint', 'hardship_flag', 'disbursement_method', 'debt_settlement_flag', 'sec_app_flag'])

    grade_model = load_model('models/subgrade_final')
    prediction = predict_model(grade_model, data_frame)[['Label','Score']]
    return f"{prediction['Label'].values[0]}: {prediction['Score'].values[0]}"


class IntData(BaseModel):
    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: float
    term: str
    installment: float
    grade: str
    sub_grade: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    pymnt_plan: str
    dti: float
    delinq_2yrs: float
    fico_range_low: float
    fico_range_high: float
    inq_last_6mths: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    initial_list_status: str
    out_prncp: float
    out_prncp_inv: float
    total_pymnt: float
    total_pymnt_inv: float
    total_rec_prncp: float
    total_rec_int: float
    total_rec_late_fee: float
    recoveries: float
    collection_recovery_fee: float
    last_pymnt_amnt: float
    last_fico_range_high: float
    last_fico_range_low: float
    collections_12_mths_ex_med: float
    application_type: str
    annual_inc_joint: float
    dti_joint: float
    verification_status_joint: str
    acc_now_delinq: float
    tot_coll_amt: float
    tot_cur_bal: float
    open_acc_6m: float
    open_act_il: float
    open_il_12m: float
    open_il_24m: float
    mths_since_rcnt_il: float
    total_bal_il: float
    il_util: float
    open_rv_12m: float
    open_rv_24m: float
    max_bal_bc: float
    all_util: float
    total_rev_hi_lim: float
    inq_fi: float
    total_cu_tl: float
    inq_last_12m: float
    acc_open_past_24mths: float
    avg_cur_bal: float
    bc_open_to_buy: float
    bc_util: float
    chargeoff_within_12_mths: float
    delinq_amnt: float
    mo_sin_old_il_acct: float
    mo_sin_old_rev_tl_op: float
    mo_sin_rcnt_rev_tl_op: float
    mo_sin_rcnt_tl: float
    mort_acc: float
    mths_since_recent_bc: float
    mths_since_recent_inq: float
    num_accts_ever_120_pd: float
    num_actv_bc_tl: float
    num_actv_rev_tl: float
    num_bc_sats: float
    num_bc_tl: float
    num_il_tl: float
    num_op_rev_tl: float
    num_rev_accts: float
    num_rev_tl_bal_gt_0: float
    num_sats: float
    num_tl_120dpd_2m: float
    num_tl_30dpd: float
    num_tl_90g_dpd_24m: float
    num_tl_op_past_12m: float
    pct_tl_nvr_dlq: float
    percent_bc_gt_75: float
    pub_rec_bankruptcies: float
    tax_liens: float
    tot_hi_cred_lim: float
    total_bal_ex_mort: float
    total_bc_limit: float
    total_il_high_credit_limit: float
    revol_bal_joint: int
    hardship_flag: str
    disbursement_method: str
    debt_settlement_flag: str
    sec_app_flag: int

@app.post("/predict-intrate")
def predict_intrate(data: IntData):
    data_frame = pd.DataFrame(
        data=np.array([data.loan_amnt, data.funded_amnt, data.funded_amnt_inv, data.term, data.installment, data.grade, data.sub_grade, data.home_ownership, data.annual_inc, data.verification_status, data.pymnt_plan, data.dti, data.delinq_2yrs, data.fico_range_low, data.fico_range_high, data.inq_last_6mths, data.open_acc, data.pub_rec, data.revol_bal, data.revol_util, data.total_acc, data.initial_list_status, data.out_prncp, data.out_prncp_inv, data.total_pymnt, data.total_pymnt_inv, data.total_rec_prncp, data.total_rec_int, data.total_rec_late_fee, data.recoveries, data.collection_recovery_fee, data.last_pymnt_amnt, data.last_fico_range_high, data.last_fico_range_low, data.collections_12_mths_ex_med, data.application_type, data.annual_inc_joint, data.dti_joint, data.verification_status_joint, data.acc_now_delinq, data.tot_coll_amt, data.tot_cur_bal, data.open_acc_6m, data.open_act_il, data.open_il_12m, data.open_il_24m, data.mths_since_rcnt_il, data.total_bal_il, data.il_util, data.open_rv_12m, data.open_rv_24m, data.max_bal_bc, data.all_util, data.total_rev_hi_lim, data.inq_fi, data.total_cu_tl, data.inq_last_12m, data.acc_open_past_24mths, data.avg_cur_bal, data.bc_open_to_buy, data.bc_util, data.chargeoff_within_12_mths, data.delinq_amnt, data.mo_sin_old_il_acct, data.mo_sin_old_rev_tl_op, data.mo_sin_rcnt_rev_tl_op, data.mo_sin_rcnt_tl, data.mort_acc, data.mths_since_recent_bc, data.mths_since_recent_inq, data.num_accts_ever_120_pd, data.num_actv_bc_tl, data.num_actv_rev_tl, data.num_bc_sats, data.num_bc_tl, data.num_il_tl, data.num_op_rev_tl, data.num_rev_accts, data.num_rev_tl_bal_gt_0, data.num_sats, data.num_tl_120dpd_2m, data.num_tl_30dpd, data.num_tl_90g_dpd_24m, data.num_tl_op_past_12m, data.pct_tl_nvr_dlq, data.percent_bc_gt_75, data.pub_rec_bankruptcies, data.tax_liens, data.tot_hi_cred_lim, data.total_bal_ex_mort, data.total_bc_limit, data.total_il_high_credit_limit, data.hardship_flag, data.disbursement_method, data.debt_settlement_flag, data.revol_bal_joint, data.sec_app_flag]).reshape(1, 97),
        columns=['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'installment', 'grade', 'sub_grade', 'home_ownership', 'annual_inc', 'verification_status', 'pymnt_plan', 'dti', 'delinq_2yrs', 'fico_range_low', 'fico_range_high', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'last_fico_range_high', 'last_fico_range_low', 'collections_12_mths_ex_med', 'application_type', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct','mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_inq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'hardship_flag', 'disbursement_method', 'debt_settlement_flag', 'revol_bal_joint', 'sec_app_flag'])

    path = 'models/int_model.joblib'
    int_model = joblib.load(path)
    #with open('models/int_model.joblib', 'rb') as path:
    #    int_model = joblib.load(path)
    prediction = int_model.predict(data_frame)
    return str(prediction)