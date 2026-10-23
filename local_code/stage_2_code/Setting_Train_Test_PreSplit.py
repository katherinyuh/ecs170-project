from local_code.base_class.setting import setting


class Setting_Train_Test_PreSplit(setting):

    def load_run_save_evaluate(self):
        # load the pre-split dataset
        loaded_data = self.dataset.load()

        # pass train/test straight to the method (no split, no CV)
        self.method.data = {
            'train': {'X': loaded_data['train']['X'], 'y': loaded_data['train']['y']},
            'test': {'X': loaded_data['test']['X'], 'y': loaded_data['test']['y']},
        }
        learned_result = self.method.run()

        # save raw predictions
        self.result.data = learned_result
        self.result.save()

        # evaluate
        self.evaluate.data = learned_result
        return self.evaluate.evaluate()