include $(ENV_FILE)
export

finney = wss://entrypoint-finney.opentensor.ai:443
testnet = wss://test.finney.opentensor.ai:443
localnet = $(LOCALNET)

ifeq ($(NETWORK),localnet)
   netuid = 1
else ifeq ($(NETWORK),testnet)
   netuid = 340
else ifeq ($(NETWORK),finney)
   netuid = 999
endif


miner:
	pm2 start --name $(MINER_NAME) python3 -- snp_oracle/miners/miner.py \
		--neuron.name $(MINER_NAME) \
		--wallet.name $(COLDKEY) \
		--wallet.hotkey $(MINER_HOTKEY) \
		--subtensor.chain_endpoint $($(NETWORK)) \
		--axon.port $(MINER_PORT) \
		--netuid $(netuid) \
		--logging.level $(LOGGING_LEVEL) \
		--timeout $(TIMEOUT) \
		--vpermit_tao_limit $(VPERMIT_TAO_LIMIT) \
		--forward_function $(FORWARD_FUNCTION) \

validator:

	# Delete pm2 processes if they're already running
	bash ./snp_oracle/validators/scripts/pm2_del.sh

	# Generate the pm2 config file
	bash ./snp_oracle/validators/scripts/create_pm2_config.sh

	pm2 start app.config.js
