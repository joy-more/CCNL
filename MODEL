class BGTGANModel(BaseModel):

    def __init__(self, opt):
        super(BGTGANModel, self).__init__(opt)
        # define network
        self.net = build_network(opt['network'])
        self.net = self.model_to_device(self.net)
        self.print_network(self.net)

        # load pretrained model
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key', 'params')
            self.load_network(self.net, load_path, self.opt['path'].get('strict_load', True), param_key)

        self.log_size = int(math.log(self.opt['network']['out_size'], 2))

        if self.is_train:
            self.init_training_settings()
