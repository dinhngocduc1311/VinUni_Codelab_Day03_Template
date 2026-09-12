"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import importlib
import json
import os
import re
import sys
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
from tools import TOOL_MAP

if sys.platform == "win32":
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if reconfigure_stdout:
        reconfigure_stdout(encoding="utf-8")

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

class ChatbotBaseline:
    """Baseline LLM Chatbot without ReAct Loop or Tools"""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        if self.api_key:
            try:
                genai = importlib.import_module("google.generativeai")
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(
                    f"Bạn là chatbot tư vấn du lịch. Hãy trả lời câu hỏi sau của khách hàng mà KHÔNG dùng tool hay internet: {user_input}"
                )
                return {
                    "status": "success",
                    "answer": response.text,
                    "tool_calls": [],
                    "mode": "live_api"
                }
            except Exception:
                pass

        return {
            "status": "success",
            "answer": f"[Chatbot Baseline] Bạn có thể tra cứu thông tin chuyến bay trên các trang của hãng hàng không và theo dõi thời tiết qua các ứng dụng dự báo. Tôi không có quyền truy cập dữ liệu thời gian thực và không sử dụng công cụ.",
            "tool_calls": [],
            "mode": "mock_baseline"
        }

class ReActAgent:
    """Production-grade ReAct Agent with Tool Registry and Safeguards"""
    def __init__(self, max_iterations: int = 5, api_key: Optional[str] = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(
            char for char in unicodedata.normalize("NFD", text.lower())
            if unicodedata.category(char) != "Mn"
        ).replace("đ", "d")

    def parse_city_code(self, text: str) -> str:
        codes = re.findall(r"\b(?:SGN|HAN|DAD)\b", text.upper())
        if codes:
            return codes[-1]
        normalized = self._normalize(text)
        for name, code in (("ha noi", "HAN"), ("ho chi minh", "SGN"),
                           ("sai gon", "SGN"), ("da nang", "DAD")):
            if name in normalized:
                return code
        return "SGN"

    def _execute(self, action: Any) -> Any:
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format"}
        if not isinstance(action, dict):
            return {"error": "Invalid action format"}
        try:
            tool_name = str(action["name"]).strip().lower()
            args = action["args"]
        except (KeyError, TypeError):
            return {"error": "Invalid action format"}
        if not isinstance(args, dict):
            return {"error": "Invalid action format"}
        tool = TOOL_MAP.get(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found in registry"}
        try:
            return tool(**args)
        except Exception as exc:
            return {"error": str(exc)}

    def _flight_args(self, user_input: str) -> Dict[str, Any]:
        codes = re.findall(r"\b(?:SGN|HAN|DAD)\b", user_input.upper())
        origin, destination = (codes[:2] + ["HAN", "SGN"])[:2]
        normalized = self._normalize(user_input)
        price = 5_000_000
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(trieu|tr|k)\b", normalized)
        if match:
            value = float(match.group(1).replace(",", "."))
            price = int(value * (1_000 if match.group(2) == "k" else 1_000_000))
        else:
            amount = re.search(r"\b\d{1,3}(?:[.,]\d{3})+\b", normalized)
            if amount:
                price = int(re.sub(r"[.,]", "", amount.group()))
        return {"origin": origin, "destination": destination, "max_price": price}

    def plan_and_execute_step(self, user_input: str, iteration: int) -> Tuple[str, bool]:
        normalized = self._normalize(user_input)
        needs_flight = any(term in normalized for term in ("chuyen bay", "ve may bay", "bay tu"))
        needs_weather = any(term in normalized for term in ("thoi tiet", "mac gi", "nhiet do", "mua"))

        if "chinh sach" in normalized or "doi tra" in normalized:
            answer = "Vinpearl hỗ trợ đổi vé theo điều kiện của hạng vé; vui lòng liên hệ nơi xuất vé để kiểm tra phí và chênh lệch giá."
            self.trace.append({"iteration": iteration, "thought": "Đây là câu hỏi FAQ, không cần dùng tool.", "final_answer": answer})
            return answer, True

        if not needs_flight and not needs_weather:
            answer = "Tôi chỉ có thể hỗ trợ tra cứu chuyến bay và thời tiết trong bài lab này."
            self.trace.append({"iteration": iteration, "thought": "Không có tool phù hợp.", "final_answer": answer})
            return answer, True

        if needs_flight and iteration == 1:
            args = self._flight_args(user_input)
            thought = f"Tra chuyến bay từ {args['origin']} đi {args['destination']} trong ngân sách."
            action = {"name": "get_flight_info", "args": args}
            observation = self._execute(action)
            self.trace.append({"iteration": iteration, "thought": thought, "action": action, "observation": observation})
            if isinstance(observation, dict) and "error" in observation:
                return f"Không thể tra cứu chuyến bay: {observation['error']}", True
            if not needs_weather:
                if not observation:
                    return f"Không tìm thấy chuyến bay phù hợp từ {args['origin']} đi {args['destination']}.", True
                flights = "\n".join(
                    f"- {flight['airline']} ({flight['flight_number']}): {flight['departure_time']} - {flight['price_vnd']:,} VNĐ"
                    for flight in observation
                )
                return flights, True
            return thought, False

        if needs_weather and (iteration == 1 or iteration == 2):
            city_code = self.parse_city_code(user_input)
            thought = f"Tra thời tiết tại {city_code}."
            action = {"name": "get_weather_forecast", "args": {"city_code": city_code}}
            observation = self._execute(action)
            self.trace.append({"iteration": iteration, "thought": thought, "action": action, "observation": observation})
            if "error" in observation:
                return f"Không thể tra cứu thời tiết: {observation['error']}", True
            if not needs_flight:
                return (
                    f"Thời tiết tại {observation.get('city', city_code)}: "
                    f"{observation.get('temperature_c', 'N/A')}°C, {observation.get('condition', '')}.\n"
                    f"Gợi ý: {observation.get('recommendation', '')}",
                    True,
                )
            return thought, False

        flight_data = next(item["observation"] for item in self.trace if item.get("action", {}).get("name") == "get_flight_info")
        weather = next(item["observation"] for item in self.trace if item.get("action", {}).get("name") == "get_weather_forecast")
        flights = "\n".join(
            f"- {flight['airline']} ({flight['flight_number']}): {flight['departure_time']} - {flight['price_vnd']:,} VNĐ"
            for flight in flight_data
        ) or "Không tìm thấy chuyến bay phù hợp."
        answer = (
            f"Thông tin chuyến bay:\n{flights}\n\n"
            f"Thời tiết tại {weather.get('city', '')}: {weather.get('temperature_c', 'N/A')}°C, "
            f"{weather.get('condition', '')}.\nGợi ý: {weather.get('recommendation', '')}"
        )
        self.trace.append({"iteration": iteration, "thought": "Đã đủ dữ liệu để trả lời.", "final_answer": answer})
        return answer, True

    def run(self, user_input: str) -> Dict[str, Any]:
        self.trace = []
        for iteration in range(1, self.max_iterations + 1):
            answer, is_final = self.plan_and_execute_step(user_input, iteration)
            if is_final:
                return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}
        return {
            "answer": "Không thể hoàn thành trong số bước tối đa.",
            "trace": self.trace,
            "iterations": self.max_iterations,
            "status": "max_iterations_reached",
        }


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"
    
    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))
    
    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()