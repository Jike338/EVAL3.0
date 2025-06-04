import re
from models.gpt import GPT
from prompts import demo_prompt_mathvista, demo_prompt_superclevr_counting
from tqdm import tqdm

class Extractor:
    def __init__(self, 
                 items_with_raw_responses, 
                 args=None, 
                 use_quick_extract_w_gpt=True, 
                 use_gpt_extract=False, 
                 use_answer_tag_extract=False):
        self.items_with_raw_responses = items_with_raw_responses
        self.args = args
        self.use_quick_extract_w_gpt = use_quick_extract_w_gpt
        self.use_gpt_extract = use_gpt_extract
        self.use_answer_tag_extract = use_answer_tag_extract
        self.gpt_extractor = GPT()
        
        if self.args.gen_prompt_suffix_type == "cot_tag":
            self.use_quick_extract_w_gpt = False
            self.use_answer_tag_extract = True
        
    def _save(self):
        from utils import save_response_to_json
        save_response_to_json(self.args, self.items_with_raw_responses)

    def _build_gpt_extraction_prompt(self, question_prompt, resp):
        demo_prompt = self.demo_prompt.strip()
        test_prompt = f"{question_prompt}\n\n{resp}"
        full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
        return full_prompt

    def _extract_with_gpt(self, question_prompt, resp):
        print("using GPT extract")
        full_prompt = self._build_gpt_extraction_prompt(question_prompt, resp)
        return self.gpt_extractor.extract_answer_from_raw_response(full_prompt)

    def _is_simple_response(self, resp):
        return (
            re.fullmatch(r"[A-Z]", resp) or              
            re.fullmatch(r"\d+(?:\.\d+)?", resp)          
        )

    def _extract_answer_content(self, resp):
        # Extract anything within <answer>...</answer> tags
        answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(answer_pattern, resp, re.DOTALL)
        
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _strip_code_fences(txt: str) -> str:
        """Remove ``` … ``` fences (with or without language tags)."""
        return re.sub(r'```[a-zA-Z]*\n?', '', txt).replace('```', '')

    # ---------- bounding‑box ----------
    def _extract_bbox(self, resp: str) -> str:
        """
        Return a clean “[x1,y1,x2,y2]” string or "none".
        Works for:
            • plain lists  [10,51,308,415]
            • floats       [8.64,55.65,331.04,422.2]
            • JSON blocks  ```json [{"bbox_2d":[…]}] ```
        """
        txt = self._strip_code_fences(resp)

        # 1) quick regex ‑– four ints / floats inside [...]
        m = re.search(
            r'\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*'
            r'([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]', txt
        )
        if m:
            g = m.groups()
            return f"[{g[0]},{g[1]},{g[2]},{g[3]}]"

        # 2) try JSON parsing for {"bbox_2d":[…]}
        try:
            # find the smallest dict that contains "bbox_2d"
            for snippet in re.findall(r'\{[^{}]*"bbox_2d"[^{}]*\}', txt, flags=re.DOTALL):
                data = json.loads(snippet)
                if "bbox_2d" in data:
                    bbox = data["bbox_2d"]
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        return f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
        except Exception:
            pass

        return "none"

    # ---------- multiple‑choice ----------
    @staticmethod
    def _extract_choices(prompt: str):
        """Return {'A':'…', 'B':'…', …} or None."""
        if "\nchoices:" not in prompt.lower():
            return None
        choices_txt = prompt.split("\nChoices:")[1]
        pairs = re.findall(r'\(([A-E])\)\s*([^\n]+)', choices_txt)
        return {l.upper(): v.strip() for l, v in pairs} if pairs else None

    # ---------- public wrapper ----------
    def _extract_mix_data(self, question_prompt: str, resp: str):
        """
        Clean up an individual model response *resp* according to *question_prompt*.
        """
        resp = resp.strip()
        qp_low = question_prompt.lower()

        # ---- 1. multiple‑choice question ----
        if "\nchoices" in qp_low:
            choices = self._extract_choices(question_prompt)
            if not choices:                        # should never happen, but be safe
                return resp

            # (a) pure letter variants   A  (A)  B:  (c) …
            m_letter = re.match(r'^\s*\(?([A-E])\)?\s*[:\-]?\s*$', resp, flags=re.I)
            if m_letter:
                return choices.get(m_letter.group(1).upper(), resp)

            # (b) inline “(A) 2” – strip marker, keep value
            m_inline = re.match(r'^\s*\([A-E]\)\s*([^\n]+)$', resp)
            if m_inline:
                return m_inline.group(1).strip()

            # (c) mixed answer – just delete “(A)” fragments
            cleaned = re.sub(r'\s*\([A-E]\)\s*', '', resp).strip()
            return choices.get(cleaned.upper(), cleaned)  # fall back to raw text

        # ---- 2. bounding‑box style question ----
        if any(key in qp_low for key in ("bounding box", "bound box", "bbox")):
            return self._extract_bbox(resp)

        # ---- 3. plain text ----
        return resp

    def _extract(self, question_prompt, raw_responses):
        extracted_responses = []
        for resp in raw_responses:
            if self.use_quick_extract_w_gpt:
                resp = resp.strip()
                if not self._is_simple_response(resp):
                    resp = self._extract_with_gpt(question_prompt, resp)
            elif self.use_answer_tag_extract:
                resp = self._extract_answer_content(resp)
            elif self.use_gpt_extract:
                resp = self._extract_with_gpt(question_prompt, resp)
            elif self.extract_mix:
                resp = self._extract_mix_data(question_prompt, resp)
            else:
                raise NotImplementedError(f"No extraction method selected.")
            extracted_responses.append(resp)
        return extracted_responses
    
    def _extract_raw_response(self):
        for item in tqdm(self.items_with_raw_responses, desc="Extracting answers"):
            if 'extracted_response' in item:
                print("already exists")
                continue
            raw_responses = item['raw_response']
            question_prompt = item['question_prompt']
            extracted_responses = self._extract(question_prompt, raw_responses)
            item['extracted_response'] = extracted_responses
    
    def extract_ans_and_save(self):
        if "mathvista" in self.args.task_name.lower():
            self.demo_prompt = demo_prompt_mathvista
        elif "superclevr_counting" in self.args.task_name.lower():
            self.demo_prompt = demo_prompt_superclevr_counting
        elif "mix_data" in self.args.task_name.lower():
            self.use_quick_extract_w_gpt = False
            self.extract_mix = True
        else:
            self.demo_prompt = demo_prompt_mathvista
        
        self._extract_raw_response()
        self.args.duty_type = "extract"
        self._save()
