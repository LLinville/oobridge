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
        self.instance_name = f"Random533256"
        self.api_key = "VvAC0CHLp3OoKyL_og5_8w"
        self.CSRF_Token = "X-CSRF-Token=a2fb4186c84e68bf674757f3fb6afa143fd57a04fef010e7b494e9cbb45f6745"
        self.cluster_headers = {"apikey": self.api_key}
        self.generator_headers = {
            "Cookie": "X-CSRF-Token=a2fb4186c84e68bf674757f3fb6afa143fd57a04fef010e7b494e9cbb45f6745",
            "X-Csrf-Token": "9d19ca67e4b69f3b6fc9aa605ff11cb8efaf5083794fe77595089b7ee1e1ecef7521422960a6d288fbea5706c34cf2e89a8cc7d1bf4a339b2d854da0fb49b14f"
        }

        logging.basicConfig(
            filename='koboldai_horde_oobridge.log',
            level=logging.INFO,
            format='%(asctime)s | %(message)s'
        )

    def run(self):
        self.running = True
        while self.running:
            job = self.get_job()
            if job is None:
                logging.info("No work yet")
                time.sleep(5)
                continue
            logging.info(f"Received job {json.dumps(job)}")
            result = self.generate(settings=job)
            logging.info(f"Job {json.dumps(job)} returned: {result}")
            self.send_results(job['id'], result)
            # Request prompt from stablehorde
            # Forward prompt to

    def get_job(self):

        gen_dict = {
            "name": self.instance_name,
            "models": ["TheBloke_Wizard-Vicuna-13B-Uncensored-GPTQ"],
            "max_length": self.max_response_tokens,
            "max_context_length": self.max_context_tokens,
            "priority_usernames": [],
            "bridge_agent": f"KoboldAI Bridge:10:https://github.com/db0/KoboldAI-Horde-Bridge",
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
        logging.debug("test")
        if settings is None:
            settings = {}

        generate_request_body = {
            "prompt": prompt,
            "max_new_tokens": min(100,self.max_response_tokens),
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
        print(f"Generated {len(generated_text)} chars (~{len(generated_text) // 4} tokens in {end_time-start_time:.3f} sec: {len(generated_text) / (end_time-start_time):.2f} chars/sec, ~{len(generated_text) / (end_time-start_time) / 4:.2f} tokens/sec")
        return json.loads(generate_response.text)['results'][0]['text']

    def send_results(self, id, generated_text):
        submit_dict = {
            "id": id,
            "generation": generated_text,
        }
        submit_response = requests.post(self.cluster_url + '/api/v2/generate/text/submit', json = submit_dict, headers = self.cluster_headers)
        logging.info(f"Send results response code: {submit_response.status_code}")
        logging.info(f"Submitted generation with id {id} and contributed for {submit_response.json()['reward']}")




    def send_failure(self):
        pass


if __name__ == "__main__":
    bridge = Bridge()
    # example_job = {"payload": {"prompt": "Alone in the woods, Tom heard a noise that sounded like " , "n": 1, "max_context_length": 2048, "max_length": 82, "rep_pen": 1.1, "rep_pen_range": 1024, "rep_pen_slope": 0.7, "temperature": 0.74, "tfs": 0.97, "top_a": 0.75, "top_k": 0, "top_p": 0.5, "typical": 0.19, "sampler_order": [6, 5, 4, 3, 2, 1, 0], "quiet": True}, "id": "63e19ed4-d612-4925-b0f9-d9fc0bc7d561", "skipped": {}, "softprompt": None, "model": "notstoic/OPT-13B-Erebus-4bit-128g"}
    # bridge.generate(settings=example_job['payload'])
    bridge.run()
