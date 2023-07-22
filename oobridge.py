import json
import time

import requests
import logging





class Bridge():
    def __init__(self):
        self.max_context_tokens = 1024
        self.max_response_tokens = 512
        self.cluster_url = "https://stablehorde.net"
        self.generator_url = "http://127.0.0.1:8000"

        # Load json containing instance_name, api_key, "Cookie": "X-CSRF-Token=...",  "X-Csrf-Token"
        self.load_credentials("credentials.json")

        self.log_request_content = True

        logging.basicConfig(
            filename='H:\ML\LLM\oobridge_logs\koboldai_horde_oobridge.log',
            level=logging.INFO,
            format='%(asctime)s | %(message)s'
        )

    def load_credentials(self, filename):
        with open(filename) as credentials_file:
            credentials = json.load(credentials_file)
        self.instance_name = credentials['instance_name']
        self.api_key = credentials['api_key']
        self.cluster_headers = {"apikey": self.api_key}
        self.generator_headers = {
            "Cookie": credentials['Cookie'],  # X-CSRF-Token=...
            "X-Csrf-Token": credentials['X-Csrf-Token']
        }

    def run(self):
        self.running = True
        while self.running:
            job = self.get_job()
            if job is None:
                logging.info("No work yet")
                time.sleep(3)
                continue

            result = self.generate(settings=job)
            if self.log_request_content:
                try:
                    logging.info(f"Job {json.dumps(job)} returned: {result}")
                except UnicodeEncodeError as ex:
                    logging.info("Failed to encode character: " + str(ex))
                except Exception as ex:
                    logging.info(ex)

            self.send_results(job['id'], result)

    def get_job(self):

        gen_dict = {
            "name": self.instance_name,
            "models": ["TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"],
            "max_length": self.max_response_tokens,
            "max_context_length": self.max_context_tokens,
            "priority_usernames": [],
            "bridge_agent": f"oobridge:1:https://github.com/LLinville/oobridge",
        }
        response = requests.post(self.cluster_url + '/api/v2/generate/text/pop', json = gen_dict, headers = self.cluster_headers)

        if not response.json().get('id'):
            return None

        payload = response.json()['payload']

        name_mapping = {
            "max_context_length": "truncation_length",
            "max_length": "max_new_tokens",
            "rep_pen": "repetition_penalty",
            "typical": "typical_p"
        }

        for cluster_name, generator_name in name_mapping.items():
            if cluster_name in payload:
                payload[generator_name] = payload[cluster_name]

        payload['id'] = response.json()['id']
        return payload

    def generate(self, prompt="", settings=None):
        if settings is None:
            settings = {}

        generate_request_body = {
            "prompt": prompt,
            "max_new_tokens": min(512, self.max_response_tokens),
            "temperature": 0.63,
            "top_p": 0.98,
            "typical_p": 1,
            "top_a": 0,
            "tfs": 1,
            "epsilon_cutoff": 0,
            "eta_cutoff": 0,
            "repetition_penalty": 1.05,
            "encoder_repetition_penalty": 1,
            "top_k": 0,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "penalty_alpha": 0,
            "length_penalty": 1,
            "early_stopping": False,
            "seed": -1,
            "add_bos_token": True,
            "stopping_strings": [],
            "truncation_length": 1024,
            "ban_eos_token": True,
            "skip_special_tokens": True,
            "do_sample": True,

        }

        for setting, value in settings.items():
            generate_request_body[setting] = value
        start_time = time.time()
        generate_response = requests.post(f"{self.generator_url}/generate_textgenerationwebui", json=generate_request_body, headers=self.generator_headers)
        end_time = time.time()
        generated_text = json.loads(generate_response.text)['results'][0]['text']
        print(f"Generated {len(generated_text)} chars (~{len(generated_text) // 4} tokens) in {end_time-start_time:.3f} sec: {len(generated_text) / (end_time-start_time):.2f} chars/sec, ~{len(generated_text) / (end_time-start_time) / 4:.2f} tokens/sec | ~{len(settings['prompt']) // 4} token context, max {settings['max_new_tokens']} new tokens")
        logging.info(f"Generated {len(generated_text)} chars (~{len(generated_text) // 4} tokens) in {end_time-start_time:.3f} sec: {len(generated_text) / (end_time-start_time):.2f} chars/sec, ~{len(generated_text) / (end_time-start_time) / 4:.2f} tokens/sec | ~{len(settings['prompt']) // 4} token context, max {settings['max_new_tokens']} new tokens")
        return json.loads(generate_response.text)['results'][0]['text']

    def send_results(self, id, generated_text):
        submit_dict = {
            "id": id,
            "generation": generated_text,
        }
        submit_response = requests.post(self.cluster_url + '/api/v2/generate/text/submit', json=submit_dict, headers=self.cluster_headers)
        logging.info(f"Send results response code: {submit_response.status_code}")
        logging.info(f"{submit_response.status_code}: Submitted generation with id {id} and contributed for {submit_response.json()['reward']}")

    def send_failure(self):
        pass


if __name__ == "__main__":
    bridge = Bridge()
    bridge.run()
