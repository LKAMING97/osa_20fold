# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:LGB_inference.py
@Time:2022/8/15 19:07

"""
import tornado.ioloop
import tornado.web
import json
import joblib
import numpy as np


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello,Welcome back")

    def post(self):
        result = list()
        data = self.request.body.decode("utf-8")
        data = json.loads(data)

        data = np.array(data["data"]).reshape(-1, 8)
        # label = np.array(data["label"])
        preds_lbl = clf.predict(data)
        preds = np.around(preds_lbl, 3)
        np.save("pred.npy", preds_lbl)
        threshold = 0.5
        for pred in preds_lbl:
            result.append(1) if pred > threshold else result.append(0)
        # auc = roc_auc_score(y_score=preds_lbl, y_true=label)
        msg = {"label": result, "probability": preds.tolist()}
        self.write(json.dumps(msg))

    def data_process(self):
        pass


if __name__ == "__main__":
    application = tornado.web.Application([(r"/", MainHandler), ])
    clf = joblib.load("model/lgb_model/all_lgbm_bst_val_auc.dat")
    application.listen(5642)  # 监听端口
    tornado.ioloop.IOLoop.instance().start()
