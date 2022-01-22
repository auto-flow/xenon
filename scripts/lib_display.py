#!/usr/bin/env python

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from liquid import Liquid
from pylab import figure
from scipy import stats

import seaborn as sns

html = (Path(__file__).parent / 'template.html').read_text()

# a - b ASC

replace_clf_binary = '''
            table = table.replace('{tn}', list[i]['tn'])
            table = table.replace('{fp}', list[i]['fp'])
            table = table.replace('{fn}', list[i]['fn'])
            table = table.replace('{tp}', list[i]['tp'])
            table = table.replace('{mcc}', list[i]['mcc'])
            table = table.replace('{se}', list[i]['se'])
            table = table.replace('{sp}', list[i]['sp'])
            table = table.replace('{acc}', list[i]['acc'])
            table = table.replace('{auc}', list[i]['auc'])
            table = table.replace('{f1}', list[i]['f1'])
'''

replace_clf_multiclass = '''
            table = table.replace('{mcc}', list[i]['mcc'])
            table = table.replace('{acc}', list[i]['acc'])

            table = table.replace('{precision_macro}', list[i]['precision_macro'])
            table = table.replace('{precision_micro}', list[i]['precision_micro'])
            table = table.replace('{precision_weighted}', list[i]['precision_weighted'])
            table = table.replace('{recall_macro}', list[i]['recall_macro'])
            table = table.replace('{recall_micro}', list[i]['recall_micro'])
            table = table.replace('{recall_weighted}', list[i]['recall_weighted'])
            table = table.replace('{f1_macro}', list[i]['f1_macro'])
            table = table.replace('{f1_micro}', list[i]['f1_micro'])
            table = table.replace('{f1_weighted}', list[i]['f1_weighted'])
            table = table.replace('{roc_auc_ovo_macro}', list[i]['roc_auc_ovo_macro'])
            table = table.replace('{roc_auc_ovo_weighted}', list[i]['roc_auc_ovo_weighted'])
            table = table.replace('{roc_auc_ovr_macro}', list[i]['roc_auc_ovr_macro'])
            table = table.replace('{roc_auc_ovr_weighted}', list[i]['roc_auc_ovr_weighted'])
'''

replace_reg = '''
            table = table.replace('{r2}', list[i]['r2'])
            table = table.replace('{mse}', list[i]['mse'])
            table = table.replace('{mae}', list[i]['mae'])
            table = table.replace('{pearsonr}', list[i]['pearsonr'])
            table = table.replace('{spearmanr}', list[i]['spearmanr'])
            table = table.replace('{kendalltau}', list[i]['kendalltau'])
'''

# colspan=5

td_clf_binary = '''
<div class="tablesdivs">
    <table border="0" class="table">
        <tr>
            <td class="name">rank</td>
            <td>{ID}</td>
            <td class="name">trial_id</td>
            <td>{trial_id}</td>
            <td class="name">estimator</td>
            <td>{estimator}</td>
            <td class="name">cost time</td>
            <td>{cost_time}</td>
        </tr>
        <tr>
            <td class="name">preprocessing</td>
            <td colspan=8>{preprocessing}</td>
        </tr>
        <tr>
            <td class="name">estimating</td>
            <td colspan=8>{estimating}</td>
        </tr>
        <tr>
            <td class="name">&nbsp;</td>
            <td class="name">tn, fp, fn, tp</td>
            <td class="name">mcc</td>
            <td class="name">se</td>
            <td class="name">sp</td>
            <td class="name">acc</td>
            <td class="name">auc</td>
            <td class="name">f1</td>
            <td class="name">loss</td>
        </tr>
        <tr>
            <td class="name">validation</td>
            <td>{tn}, {fp}, {fn}, {tp} </td>
            <td>{mcc}</td>
            <td>{se}</td>
            <td>{sp}</td>  
            <td>{acc}</td>
            <td>{auc}</td>
            <td>{f1}</td>
            <td>{loss}</td>
        </tr>
    </table>
    <table border="0" class="table" cellpadding="0" cellspacing="0">
        {png_base64_list}
    </table>
</div>
'''

td_clf_multiclass = '''
<div class="tablesdivs">
    <table border="0" class="table">
        <tr>
            <td class="name">rank</td>
            <td>{ID}</td>
            <td class="name">trial_id</td>
            <td>{trial_id}</td>
            <td class="name">estimator</td>
            <td>{estimator}</td>
            <td class="name">cost time</td>
            <td>{cost_time}</td>
        </tr>
        <tr>
            <td class="name">preprocessing</td>
            <td colspan=8>{preprocessing}</td>
        </tr>
        <tr>
            <td class="name">estimating</td>
            <td colspan=8>{estimating}</td>
        </tr>
        <tr>
            <td class="name">&nbsp;</td>
            <td class="name">mcc</td>
            <td class="name">acc</td>
            <td class="name">roc_auc_ovo_macro</td>
            <td class="name">roc_auc_ovo_weighted</td>
            <td class="name">roc_auc_ovr_macro</td>
            <td class="name">roc_auc_ovr_weighted</td>
            <td class="name">loss</td>
        </tr>
        <tr>
            <td class="name">validation</td>
            <td>{mcc}</td>
            <td>{acc}</td>
            <td>{roc_auc_ovo_macro}</td>
            <td>{roc_auc_ovo_weighted}</td>  
            <td>{roc_auc_ovr_macro}</td>
            <td>{roc_auc_ovr_weighted}</td>
            <td>{loss}</td>
        </tr>
        <tr>
            <td class="name">&nbsp;</td>
            <td class="name">precision_macro</td>
            <td class="name">precision_micro</td>
            <td class="name">precision_weighted</td>
            <td class="name">recall_macro</td>
            <td class="name">recall_micro</td>  
            <td class="name">recall_weighted</td>
            <td class="name">f1_macro</td>
            <td class="name">f1_micro</td>
            <td class="name">f1_weighted</td>
        </tr>
        <tr>
            <td class="name">&nbsp;</td>
            <td>{precision_macro}</td>
            <td>{precision_micro}</td>
            <td>{precision_weighted}</td>
            <td>{recall_macro}</td>
            <td>{recall_micro}</td>  
            <td>{recall_weighted}</td>
            <td>{f1_macro}</td>
            <td>{f1_micro}</td>
            <td>{f1_weighted}</td>
        </tr>
    </table>
    <table border="0" class="table" cellpadding="0" cellspacing="0">
        {png_base64_list}
    </table>
</div>
'''

td_reg = '''
<div class="tablesdivs">
    <table border="0" class="table" cellpadding="0" cellspacing="0">
        <tr>
            <td class="name">rank</td>
            <td>{ID}</td>
            <td class="name">trial_id</td>
            <td>{trial_id}</td>
            <td class="name">estimator</td>
            <td>{estimator}</td>
            <td class="name">cost_time</td>
            <td>{cost_time}</td>
        </tr>
        <tr>
            <td class="name">preprocessing</td>
            <td colspan=7>{preprocessing}</td>
        </tr>
        <tr>
            <td class="name">estimating</td>
            <td colspan=7>{estimating}</td>
        </tr>
        <tr>
            <td class="name">&nbsp;</td>
            <td class="name">r2</td>
            <td class="name">mse</td>
            <td class="name">mae</td>
            <td class="name">pearsonr</td>
            <td class="name">spearmanr</td>
            <td class="name">kendalltau</td>
            <td class="name">loss</td>
        </tr>
        <tr>
            <td class="name">validation</td>
            <td>{r2}</td>
            <td>{mse}</td>
            <td>{mae}</td>
            <td>{pearsonr}</td>
            <td>{spearmanr}</td>
            <td>{kendalltau}</td>
            <td>{loss}</td>
        </tr>
    </table>
    <table border="0" class="table" cellpadding="0" cellspacing="0">
        {png_base64_list}
    </table>
</div>
'''

clf_binary = '''
<option>tn</option>
<option>fp</option>
<option>fn</option>
<option>tp</option>
<option>mcc</option>
<option>se</option>
<option>sp</option>
<option>acc</option>
<option>auc</option>
<option>f1</option>
<option>loss</option>
'''

clf_multiclass = '''
<option>mcc</option>
<option>acc</option>
<option>loss</option>
<option>precision_macro</option>
<option>precision_micro</option>
<option>precision_weighted</option>
<option>recall_macro</option>
<option>recall_micro</option>
<option>recall_weighted</option>
<option>f1_macro</option>
<option>f1_micro</option>
<option>f1_weighted</option>
<option>roc_auc_ovo_macro</option>
<option>roc_auc_ovo_weighted</option>
<option>roc_auc_ovr_macro</option>
<option>roc_auc_ovr_weighted</option>
'''

reg = '''
<option>r2</option>
<option>mse</option>
<option>mae</option>
<option>pearsonr</option>
<option>spearmanr</option>
<option>kendalltau</option>
<option>loss</option>
'''

box = '''
<span class="orange"></span> positive
<span class="blue"></span> negative
'''


def clf_plot(y_true, y_score, threshold=0.5, title=None):
    colors = ['#E69F00', '#56B4E9']
    names = ['positive', 'negative']
    pred_result = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    df_pos = pred_result.loc[pred_result.y_true == 1]
    df_neg = pred_result.loc[pred_result.y_true == 0]
    plt.hist([df_pos.y_score, df_neg.y_score], bins=20,
             color=colors, label=names, alpha=0.8, rwidth=0.9)
    plt.xlim(0, 1)
    plt.xlabel('prob')
    plt.ylabel('count')
    if title is not None:
        plt.title(title, fontsize=14)
    plt.axvline(x=threshold, ymax=0.95, linestyle='--', color='crimson', alpha=0.5)
    plt.grid(alpha=0.3, linestyle='--')

    buf = io.BytesIO()
    plt.savefig(buf, dpi=60, bbox_inches='tight')
    plt.clf()
    return base64.b64encode(buf.getvalue()).decode('utf8')


def confusion_matrix_plot(cf_matrix, title=None):
    ax = plt.axes()
    if title is not None:
        ax.set_title(title, fontsize=14)

    cmap = sns.color_palette("flare", as_cmap=True)
    sns.heatmap(cf_matrix, cmap=cmap, annot=True, ax=ax)
    ax.set_ylabel('True Class')
    ax.set_xlabel('Predicted Class')

    buf = io.BytesIO()
    plt.savefig(buf, dpi=60, bbox_inches='tight')
    plt.clf()
    return base64.b64encode(buf.getvalue()).decode('utf8')


def reg_plot(y_true, y_pred, xmin, xmax, title=None):
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    def score(i):
        return ("+" if i > 0 else "") + str(i)

    line = slope * y_true + intercept
    y_true_above_line = y_true[y_pred > line]
    y_pred_above_line = y_pred[y_pred > line]
    if len(y_true_above_line) == 0 or len(y_pred_above_line) == 0:
        r_value_new = 0
    else:
        slope_new, intercept_new, r_value_new, _, _ = stats.linregress(
            y_true_above_line, y_pred_above_line)
    f = figure()
    ax = f.add_subplot(111)
    plt.xlim(xmin, xmax)
    plt.plot(y_true, y_pred, 'o', y_true, line)
    plt.grid(alpha=0.3, linestyle='--')
    plt.xlabel('y_label')
    plt.ylabel('y_pred')
    if title is not None:
        plt.title(title, fontsize=14)
    text_message = 'y = ' + str(round(slope, 5)) + 'x ' + \
                   score(round(intercept, 5)) + '\n$r^2$: ' + str(round(r_value ** 2, 5)
                                                                  ) + '\n$r^2$ above line: ' + str(
        round(r_value_new ** 2, 5))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax.text(0.05, 0.95, text_message, transform=ax.transAxes,
            fontsize=14, verticalalignment='top', bbox=props)

    buf = io.BytesIO()
    plt.savefig(buf, dpi=60, bbox_inches='tight')
    plt.clf()
    return base64.b64encode(buf.getvalue()).decode('utf8')


def delete_lines(txt, do_it, contain='preprocess'):
    if not do_it:
        return txt
    ans = "\n".join([
        line for line in txt.split("\n") \
        if not contain in line])
    return ans


# main
# todo: n_columns 做成一个环境变量
def display(data: dict, n_columns=3) -> str:
    """

    Parameters
    ----------
    data:dict
    have 3 fields:
    mainTask: classification or regression
    subTask: binary or multiclass
    records: List[dict], trial records
    y_train: train target label

    Returns
    -------
    html:str
    Generated html content

    """
    mainTask = data["mainTask"]
    subTask = data["subTask"]
    # binary: 二分类
    # multiclass ：多分类
    records = data["records"]
    y_train = data["y_train"]
    is_xenon4bigdata = False
    info_list = list()
    for record in records:
        info = dict()
        info["trial_id"] = record["trial_id"]
        info["estimator"] = record["estimator"]
        info["cost_time"] = str("{:.4f}").format(record["cost_time"])
        try:
            info["preprocessing"] = str(record["dict_hyper_param"]["preprocessing"])
            info["estimating"] = str(record["dict_hyper_param"]["estimating"][info["estimator"]])
        except:
            # fixme: 这段代码用于适配xenon4bigdata
            from neutron.hdl import layering_config
            import pandas as pd
            from math import ceil

            is_xenon4bigdata = True
            # html表格分为3列
            # info["preprocessing"] = record.get('preprocessing', '')
            estimator_dict = layering_config(record.get('dict_hyper_param', ''))['estimator']

            params = estimator_dict[list(estimator_dict.keys())[0]]
            data = [(k, v) for k, v in params.items()]
            n_rows = ceil(len(data) / n_columns)  # 向上取整
            df_data = np.zeros([n_rows, n_columns * 2], dtype=object)
            for i in range(n_rows * n_columns):
                if i < len(data):
                    param, value = data[i]
                else:
                    param, value = '', ''
                r, c = i % n_rows, i // n_rows
                df_data[r, c * 2] = param
                df_data[r, c * 2 + 1] = value
            columns = []
            param_columns = []
            for i in range(n_columns):
                columns += [f'param{i}', f'value{i}']
                param_columns += [f'param{i}']
            df = pd.DataFrame(df_data)  # , columns=columns
            params_keys = set(list(params.keys()))

            def highlight_param(val):
                '''http://xh.5156edu.com/page/z1015m9220j18754.html'''
                # fixme: 觉得不好看就自己折腾吧
                if val in params_keys:
                    return 'background-color: #E0FFFF'
                else:
                    if val:
                        return 'background-color: #FFFACD'
                    else:
                        return ''

            # fixme： 我服了pandas这个可视化了，暂时这样，不浪费时间了
            df.columns = [' ' * i for i in range(n_columns * 2)]
            styler = df.style.applymap(highlight_param)
            styler.hide_index()
            # styler.columns = []
            df_html = styler.render(index=False, header=False)
            info["estimating"] = df_html
        info["loss"] = str("{:.4f}").format(record["loss"])

        if mainTask == "classification":  # fixme: 如果不是二分类就报错
            try:
                cms = record["additional_info"]["confusion_matrices"]
                cm = np.concatenate(np.sum(cms, axis=0)) / len(cms)
            except:
                cm = [0] * 4
            info["mcc"] = str("{:.4f}").format(record["all_score"]["mcc"])
            info["acc"] = str("{:.4f}").format(record["all_score"]["accuracy"])
            if subTask == "binary":
                info["tn"] = str(cm[0])
                info["fp"] = str(cm[1])
                info["fn"] = str(cm[2])
                info["tp"] = str(cm[3])
                info["se"] = str("{:.4f}").format(record["all_score"]["sensitivity"])
                info["sp"] = str("{:.4f}").format(record["all_score"]["specificity"])
                info["auc"] = str("{:.4f}").format(record["all_score"].get("auc", -1))
                info["f1"] = str("{:.4f}").format(record["all_score"]["f1"])
            else:  # multiclass
                info["precision_macro"] = str("{:.4f}").format(record["all_score"]["precision_macro"])
                info["precision_micro"] = str("{:.4f}").format(record["all_score"]["precision_micro"])
                info["precision_weighted"] = str("{:.4f}").format(record["all_score"]["precision_weighted"])

                info["recall_macro"] = str("{:.4f}").format(record["all_score"]["recall_macro"])
                info["recall_micro"] = str("{:.4f}").format(record["all_score"]["recall_micro"])
                info["recall_weighted"] = str("{:.4f}").format(record["all_score"]["recall_weighted"])

                info["f1_macro"] = str("{:.4f}").format(record["all_score"]["f1_macro"])
                info["f1_micro"] = str("{:.4f}").format(record["all_score"]["f1_micro"])
                info["f1_weighted"] = str("{:.4f}").format(record["all_score"]["f1_weighted"])

                info["roc_auc_ovo_macro"] = str("{:.4f}").format(record["all_score"]["roc_auc_ovo_macro"])
                info["roc_auc_ovo_weighted"] = str("{:.4f}").format(record["all_score"]["roc_auc_ovo_weighted"])
                info["roc_auc_ovr_macro"] = str("{:.4f}").format(record["all_score"]["roc_auc_ovr_macro"])
                info["roc_auc_ovr_weighted"] = str("{:.4f}").format(record["all_score"]["roc_auc_ovr_weighted"])

        else:  # regression
            info["r2"] = str("{:.4f}").format(record["all_score"]["r2"])
            info["mse"] = str("{:.4f}").format(record["all_score"]["mean_squared_error"])
            info["mae"] = str("{:.4f}").format(record["all_score"]["mean_absolute_error"])
            info["pearsonr"] = str("{:.4f}").format(record["all_score"]["pearsonr"])
            info["spearmanr"] = str("{:.4f}").format(record["all_score"]["spearmanr"])
            info["kendalltau"] = str("{:.4f}").format(record["all_score"]["kendalltau"])

        info["img"] = list()
        cv = len(record["y_info"]["y_true_indexes"])
        y_true_all = list()
        y_score_all = list()
        y_pred_all = list()

        for i in range(cv):
            index_list = record["y_info"]["y_true_indexes"][i]
            y_true = y_train[index_list]
            y_true_all.extend(list(y_true))
            title = "valid " + str(i + 1)
            if mainTask == "classification":
                if subTask == "binary":
                    y_score = record["y_info"]['y_preds'][i][:, 1]
                    y_score_all.extend(list(y_score))
                    info["img"].append(
                        clf_plot(
                            y_true,
                            y_score,
                            title=title
                        )
                    )
                else:  # multiclass
                    # only plot confusion matrix
                    info["img"].append(
                        confusion_matrix_plot(
                            cms[i],
                            title
                        )
                    )
            else:  # regression
                y_pred = record["y_info"]['y_preds'][i]
                y_pred_all.extend(list(y_pred))
                info["img"].append(
                    reg_plot(
                        y_true,
                        y_pred,
                        min(y_train),
                        max(y_train),
                        title=title
                    )
                )
        # cvmix
        if mainTask == "classification":
            if subTask == "binary":
                info["img"].append(
                    clf_plot(
                        y_true_all,
                        y_score_all,
                        title="all validation"
                    )
                )
            else:  # multiclass
                # TODO: plot heatmap
                pass
        else:  # regression
            info["img"].append(
                reg_plot(
                    y_true_all,
                    y_pred_all,
                    min(y_train),
                    max(y_train),
                    title="all validation"
                )
            )

        info_list.append(info)

    if mainTask == "classification":
        if subTask == "binary":
            temp_dict = {
                "list": info_list,
                "select": clf_binary,
                "replace": replace_clf_binary,
                "metatable": delete_lines(td_clf_binary, is_xenon4bigdata),
                "box": box,
            }
        else:  # multiclass
            temp_dict = {
                "list": info_list,
                "select": clf_multiclass,
                "replace": replace_clf_multiclass,
                "metatable": delete_lines(td_clf_multiclass, is_xenon4bigdata),
                "box": "",
            }

    else:  # regression
        temp_dict = {
            "list": info_list,
            "select": reg,
            "replace": replace_reg,
            "metatable": delete_lines(td_reg, is_xenon4bigdata),
            "box": "",
        }
    # fixme: xenon4bigdata preprocessing 不方便可视化
    # todo:  其实xenon4bigdata preprocessing也可以可视化，比如说特征筛选的参数，删选后特征的数量，留给后人去做
    res_html = Liquid(delete_lines(html, is_xenon4bigdata)).render(**temp_dict)

    return res_html


if __name__ == '__main__':
    from pathlib import Path
    from joblib import load

    cwd = Path(__file__).parent
    savedpath = f'{cwd}/savedpath'
    Path(savedpath).mkdir(parents=True, exist_ok=True)
    for file_name in "search_binary_digits.bz2  search_middle_digits.bz2  search_origin_digits.bz2  search_small_digits.bz2".split():
        print(file_name)
        bz_path = f'{cwd}/mock_data/{file_name}'
        data = load(bz_path)
        html_file_name = file_name.split('.')[0] + ".html"
        html_path = savedpath + "/" + html_file_name
        Path(html_path).write_text(display(data))
