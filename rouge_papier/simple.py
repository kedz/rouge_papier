import os
import pkg_resources
from subprocess import check_output, CalledProcessError
import pandas as pd
import re

from . import util


def to_dataframe(hypotheses, references, ngrams=None, length=None, 
                 length_unit="word", stem=False, remove_stopwords=False,
                 input_type="texts",
                 print_output=False, print_args=False,
                 rouge_w_weight=None):

    AVG_R_PATT = r"X ROUGE-{} Average_R: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"
    AVG_P_PATT = r"X ROUGE-{} Average_P: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"
    AVG_F_PATT = r"X ROUGE-{} Average_F: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"

    AVG_R_WPATT = r"X ROUGE-W-[\d\.]+ Average_R: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"
    AVG_P_WPATT = r"X ROUGE-W-[\d\.]+ Average_P: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"
    AVG_F_WPATT = r"X ROUGE-W-[\d\.]+ Average_F: (.*?) \(95%-conf.int. (.*?) - (.*?)\)"


    rouge_path = pkg_resources.resource_filename(
        'rouge_papier', os.path.join('data', 'ROUGE-1.5.5.pl'))
    rouge_data_path = pkg_resources.resource_filename(
        'rouge_papier', 'rouge_data')

    #-n 4 -m -a -l 100 -x -c 95
    #-r 1000 -f A -p 0.5 -t 0
 #   args = ["perl", rouge_path, "-H"]

#    print(check_output(args).decode("utf8"))

    args = ["perl", rouge_path, "-e", rouge_data_path, "-a"]

    if ngrams is not None:
        args.extend(["-n", str(ngrams)])

    if length is not None:
        if length_unit == "word":
            args.extend(["-l", str(length)])
        elif length_unit == "byte":
            args.extend(["-b", str(length)])
        else:
            raise Exception(
                ("length_unit must be either 'word' or 'byte' "
                 "but found {}").format(length_unit))
    if stem:
        args.append("-m")

    if remove_stopwords:
        args.append("-s")
    
    if rouge_w_weight is not None:
        args.extend(["-w", str(rouge_w_weight)])

    with util.TempFileManager() as manager:
        input_paths = []
        for hyp, refs in zip(hypotheses, references):
            if input_type == "texts":
                hyp_path = manager.create_temp_file(hyp)
                ref_paths = manager.create_temp_files(refs)
                input_paths.append([hyp_path, ref_paths])
            elif input_type == "paths":
                input_paths.append([str(hyp), [str(x) for x in refs]])
            else:
                raise ValueError("input_type must be 'texts' or 'paths'")
        config_path = manager.create_temp_file(
            util.make_simple_config_text(input_paths))

        try:
            args += ["-z", "SPL", config_path]
            if print_args:
                print(args)
            output = check_output(args).decode("utf8")
            if print_output:
                print(output)

        except CalledProcessError as e:
            print(e.output)
            raise e

        groups = []
        cols = []
        data = []
        if ngrams is not None:
            for ngram in range(1, ngrams + 1):
                avg_r = float(
                    re.findall(AVG_R_PATT.format(ngram), output)[0][0])
                avg_p = float(
                    re.findall(AVG_P_PATT.format(ngram), output)[0][0])
                avg_f = float(
                    re.findall(AVG_F_PATT.format(ngram), output)[0][0])
                data.extend([avg_r, avg_p, avg_f])
                cols.extend(["Recall", "Precision", "F-Score"])
                groups.extend(["ROUGE-{}".format(ngram)] * 3)
                
        avg_r = float(
            re.findall(AVG_R_PATT.format("L"), output)[0][0])
        avg_p = float(
            re.findall(AVG_P_PATT.format("L"), output)[0][0])
        avg_f = float(
            re.findall(AVG_F_PATT.format("L"), output)[0][0])
        data.extend([avg_r, avg_p, avg_f])
        cols.extend(["Recall", "Precision", "F-Score"])
        groups.extend(["ROUGE-L"] * 3)

        if rouge_w_weight is not None: 
            avg_r = float(
                re.findall(AVG_R_WPATT, output)[0][0])
            avg_p = float(
                re.findall(AVG_P_WPATT, output)[0][0])
            avg_f = float(
                re.findall(AVG_F_WPATT, output)[0][0])
            data.extend([avg_r, avg_p, avg_f])
            cols.extend(["Recall", "Precision", "F-Score"])
            groups.extend(["ROUGE-W"] * 3)

        return pd.DataFrame([data], index=["avg."],
            columns=pd.MultiIndex.from_tuples(zip(groups,cols)))
