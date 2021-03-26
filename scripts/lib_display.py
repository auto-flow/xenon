#!/usr/bin/env python

import base64
import io
import pickle
import os
import numpy as np
import pandas as pd

try:
    from liquid import Liquid
except:
    os.system("pip install liquidpy==0.5.0")
    from liquid import Liquid
import matplotlib.pyplot as plt
from pylab import figure
from scipy import stats

html = '''
<html>
<head>
    <title>searching_record</title>
    <style type="text/css">
        .fixed {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            padding: 10px 20px;
            background-color: #ddd;
        }
        .tablesdivs {
          border: 1px solid #999;
          margin-bottom: 20px;
        }
        .name {
          font-weight: bold;
          width: 100px;
          padding: 7px 10px;
        }

        .orange, 
        .blue {
            display: inline-block;
            width: 30px;
            height: 20px;
            margin-left: 30px;
            margin-right: 5px;
            border-radius: 4px;
            vertical-align: middle;
        }

        .orange {
            background-color: #E69F00;
        } 

        .blue {
            background-color: #56B4E9;
        }

        .name:first-child {
            width: 190px;
        }

        .name {
            font-weight: bold;
            padding: 7px 10px;
        }

        td {
            width: 160px;
        }

        img {
            display:block;
            width:300px;
            height:225px;
        }
    </style>
</head>
<body>

<div class="fixed">
<select id="select_a">

    <option>trial_id</option>

{{select}}
</select>
    <select id="select_b">
        <option>ASC</option>
        <option>DESC</option>
    </select>
{{box}}

</div>
<h1 style="margin-top: 60px;">Model List</h1>
<div id="tableBox" style="margin-top: 20px;"></div>


<script>
    var list = {{list}}

    function generateTable(sortBy='trial_id', orderBy='ASC') {
        list.sort(function (a, b) {
            if (!sortBy) return 1
            if (orderBy === 'ASC') {
                return a[sortBy] - b[sortBy]
            } else {
                return b[sortBy] - a[sortBy]
            } 
        })

        var tableBoxEL = document.getElementById('tableBox')
        var oldTableEl = document.getElementById('tableList')
        var newTable = `<div id="tableList">{table}</div>`

        var tableList = ``
        var metaTable = `{{metatable}}`
        for (var i = 0; i < list.length; ++i) {
            var table = metaTable
            if(order == 'ASC'){
                table = table.replace('{ID}', i+1)
            }
            else{
                table = table.replace('{ID}', list.length - i)
            }
            table = table.replace(RegExp('{trial_id}', 'g'), list[i]['trial_id'])

            var png_base64_list = ``
            var len = list[i]['img'].length
            for (var ii = 0; ii < Math.ceil(len / 6); ++ii) {
                png_base64_list = png_base64_list.concat(`<tr>`)
                if(ii + 1 == Math.ceil(len / 6)){
                    for (var iii = ii*6; iii < len; ++iii){
                        png_base64_list = png_base64_list.concat(`<td><img src="data:image/png;base64,`, list[i]['img'][iii], `"></td>`)
                    }
                }
                else{
                    for (var iii = ii*6; iii < (ii+1)*6; ++iii){
                        png_base64_list = png_base64_list.concat(`<td><img src="data:image/png;base64,`, list[i]['img'][iii], `"></td>`)
                    }
                }
                png_base64_list = png_base64_list.concat(`</tr>`)
            }
            table = table.replace('{png_base64_list}', png_base64_list)

            table = table.replace('{estimator}', list[i]['estimator'])
            table = table.replace('{cost_time}', list[i]['cost_time'])
            table = table.replace('{preprocessing}', list[i]['preprocessing'])
            table = table.replace('{estimating}', list[i]['estimating'])
            {{replace}}
            table = table.replace('{loss}', list[i]['loss'])
            tableList += table
        }

        newTable = newTable.replace('{table}', tableList)

        oldTableEl && tableBoxEL.removeChild(oldTableEl)
        tableBoxEL.innerHTML = newTable
    }

    var sort = 'trial_id'
    var order = 'ASC'
    generateTable()
    var selectA = document.getElementById("select_a")
    var selectB = document.getElementById("select_b")
    selectA.addEventListener("change", function () {
            sort = this.value
            generateTable(sortBy=sort, orderBy=order)
        }, false)
        selectB.addEventListener("change", function () {
            order = this.value 
            generateTable(sortBy=sort, orderBy=order)
        }, false)
</script>
</body>
</html>
'''
# a - b ASC

replace_clf = '''
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

replace_reg = '''
            table = table.replace('{r2}', list[i]['r2'])
            table = table.replace('{mse}', list[i]['mse'])
            table = table.replace('{mae}', list[i]['mae'])
            table = table.replace('{pearsonr}', list[i]['pearsonr'])
            table = table.replace('{spearmanr}', list[i]['spearmanr'])
            table = table.replace('{kendalltau}', list[i]['kendalltau'])
'''

# colspan=5

td_clf = '''
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

clf = '''
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


# main
def display(data: dict) -> str:
    """

    Parameters
    ----------
    data:dict
    have 3 fields:
    mainTask: classification or regression
    records: List[dict], trial records
    y_train: train target label

    Returns
    -------
    html:str
    Generated html content

    """
    mainTask = data["mainTask"]
    records = data["records"]
    y_train = data["y_train"]

    info_list = list()
    for record in records:
        info = dict()
        info["trial_id"] = record["trial_id"]
        info["estimator"] = record["estimator"]
        info["cost_time"] = str("{:.4f}").format(record["cost_time"])
        if record["dict_hyper_param"]:
            info["preprocessing"] = str(record["dict_hyper_param"]["preprocessing"])
            info["estimating"] = str(record["dict_hyper_param"]["estimating"][info["estimator"]])
        else:
            info["preprocessing"] = record.get('preprocessing', '')
            info["estimating"] = record.get('estimating', '')
        info["loss"] = str("{:.4f}").format(record["loss"])

        if mainTask == "classification":  # fixme: 如果不是二分类就报错
            try:
                cms = record["additional_info"]["confusion_matrices"]
                cm = np.concatenate(np.sum(cms, axis=0)) / len(cms)
            except:
                cm = [0] * 4
            info["tn"] = str(cm[0])
            info["fp"] = str(cm[1])
            info["fn"] = str(cm[2])
            info["tp"] = str(cm[3])
            info["mcc"] = str("{:.4f}").format(record["all_score"]["mcc"])
            info["se"] = str("{:.4f}").format(record["all_score"]["sensitivity"])
            info["sp"] = str("{:.4f}").format(record["all_score"]["specificity"])
            info["acc"] = str("{:.4f}").format(record["all_score"]["accuracy"])
            info["auc"] = str("{:.4f}").format(record["all_score"]["roc_auc"])
            info["f1"] = str("{:.4f}").format(record["all_score"]["f1"])
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
                y_score = record["y_info"]['y_preds'][i][:, 1]
                y_score_all.extend(list(y_score))
                info["img"].append(
                    clf_plot(
                        y_true,
                        y_score,
                        title=title
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
            info["img"].append(
                clf_plot(
                    y_true_all,
                    y_score_all,
                    title="all validation"
                )
            )
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
        temp_dict = {
            "list": info_list,
            "select": clf,
            "replace": replace_clf,
            "metatable": td_clf,
            "box": box,
        }

    else:  # regression
        temp_dict = {
            "list": info_list,
            "select": reg,
            "replace": replace_reg,
            "metatable": td_reg,
            "box": "",
        }

    res_html = Liquid(html).render(**temp_dict)

    return res_html
