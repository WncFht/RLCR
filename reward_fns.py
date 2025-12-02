import re
import string

from math_verify import parse, verify


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def format_reward(format_pattern, completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    if format_pattern == "tbac":
        pattern = r".*?</think>\s*<analysis>.*?</analysis>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    elif format_pattern == "ta":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*\Z"
    elif format_pattern == "tac":
        pattern = (
            r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
        )
    elif format_pattern == "tabc":
        pattern = r".*?</think>\s*<answer>.*?</answer>\s*<analysis>.*?</analysis>\s*<confidence>.*?</confidence>\s*\Z"
    confidence_pattern = r"<confidence>(.*?)</confidence>"

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
    matches = [1.0 if match else 0.0 for match in matches]

    # if it matches, check if the confidence is between 0 and 1
    for i, match in enumerate(matches):
        if match:
            content = completion_contents[i]
            if "c" in format_pattern:
                confidence_matches = re.findall(
                    confidence_pattern, content, re.DOTALL | re.MULTILINE
                )  # Get all <confidence>...</confidence> occurrences
                last_confidence = (
                    confidence_matches[-1] if confidence_matches else ""
                )  # Get the last confidence, if exists
                if last_confidence == "":
                    matches[i] = 0.0
                else:
                    try:
                        confidence = float(last_confidence)
                        if confidence < 0 or confidence > 1:
                            matches[i] = 0.0
                        else:
                            matches[i] = 1

                    except:
                        matches[i] = 0.0
    return matches


def _split_completion_by_last_answer(content: str):
    """
    将 completion 按照最后一个 <answer>...</answer> 切分，
    返回 (含 think+answer 的前半部分, 剩余部分)。
    若无法找到闭合的 answer，则返回 (None, None)。
    """
    matches = list(re.finditer(r"<answer>[\s\S]*?</answer>", content, re.DOTALL))
    if not matches:
        return None, None
    last = matches[-1]
    before = content[: last.end()]
    after = content[last.end() :]
    return before, after


def format_answer_segment_reward(format_pattern, completions, **kwargs):
    """只校验 think+answer 片段的格式。"""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    answer_regex = re.compile(
        r".*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*\Z",
        re.DOTALL | re.IGNORECASE,
    )
    for content in completion_contents:
        before, _ = _split_completion_by_last_answer(content)
        if before is None:
            rewards.append(0.0)
        else:
            rewards.append(1.0 if re.match(answer_regex, before) else 0.0)
    return rewards


def format_confidence_segment_reward(format_pattern, completions, **kwargs):
    """只校验 analysis+confidence（或仅 confidence）片段的格式，并检查数值范围。"""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    conf_regex = re.compile(
        r"\s*(?:<analysis>[\s\S]*?</analysis>\s*)?<confidence>[\s\S]*?</confidence>\s*\Z",
        re.DOTALL | re.IGNORECASE,
    )
    confidence_pattern = re.compile(
        r"<confidence>(.*?)</confidence>", re.DOTALL | re.IGNORECASE
    )
    for content in completion_contents:
        _, after = _split_completion_by_last_answer(content)
        if after is None or not re.match(conf_regex, after):
            rewards.append(0.0)
            continue
        matches = confidence_pattern.findall(after)
        last_conf = matches[-1].strip() if matches else ""
        try:
            value = float(last_conf)
            rewards.append(1.0 if 0.0 <= value <= 1.0 else 0.0)
        except ValueError:
            rewards.append(0.0)
    return rewards


def accuracy_reward(format_pattern, completions, answer, source=None, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    ans_pattern = r"<answer>(.*?)</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer]
    matches = []
    format_rewards = format_reward(format_pattern, completions)

    for content, e, fr in zip(completion_contents, eval_contents, format_rewards):
        if fr == 0:
            matches.append(0)
        else:
            ans_matches = re.findall(
                ans_pattern, content, re.DOTALL | re.MULTILINE
            )  # Get all <answer>...</answer> occurrences
            last_answer = (
                ans_matches[-1] if ans_matches else ""
            )  # Get the last answer, if exists
            # if source exists in key and is equal to hotpot, then use the exact match score
            if source is not None and source[0] == "hotpot":
                label = exact_match_score(last_answer, e)
            else:
                attempt = parse(last_answer)
                label = verify(e, attempt)
            matches.append(float(label))
    return matches


def brier_reward(format_pattern, completions, answer, source=None, **kwargs):
    """Reward function that checks if the completion is correct."""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = []
    correctness_rewards = accuracy_reward(format_pattern, completions, answer, source)
    format_rewards = format_reward(format_pattern, completions)
    for content, cr, fr in zip(
        completion_contents, correctness_rewards, format_rewards
    ):
        if fr == 0:
            matches.append(0)
        else:
            # extract the confidence and give the reward as brier score
            confidence_matches = re.findall(
                confidence_pattern, content, re.DOTALL | re.MULTILINE
            )  # Get all <confidence>...</confidence> occurrences
            last_confidence = (
                confidence_matches[-1] if confidence_matches else ""
            )  # Get the last confidence, if exists
            if last_confidence == "":
                matches.append(0)
            else:
                try:
                    conf = float(last_confidence)
                    reward = 1 - (cr - conf) ** 2
                    matches.append(reward)
                except:
                    print(
                        "Could not parse confidence: ",
                        last_confidence,
                        "Something might be wrong",
                    )
                    matches.append(0)
    return matches


def mean_confidence_reward(completions, answer, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer]
    matches = []

    for content, e in zip(completion_contents, eval_contents):
        confidence_matches = re.findall(
            confidence_pattern, content, re.DOTALL | re.MULTILINE
        )  # Get all <confidence>...</confidence> occurrences
        last_confidence = (
            confidence_matches[-1] if confidence_matches else ""
        )  # Get the last confidence, if exists
        if last_confidence == "":
            matches.append(0.0)
        else:
            try:
                confidence = float(last_confidence)
                # clip confidence to be between 0 and 1
                confidence = max(0.0, min(confidence, 1.0))
            except:
                confidence = 0.0
            matches.append(confidence)
    return matches


def confidence_one_or_zero(completions, answer, **kwargs):
    """Reward function that extracts the last occurrence of text inside the answer tags and then checks if a label is present there"""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    eval_contents = [e for e in answer]
    matches = []

    for content, e in zip(completion_contents, eval_contents):
        confidence_matches = re.findall(
            confidence_pattern, content, re.DOTALL | re.MULTILINE
        )  # Get all <confidence>...</confidence> occurrences
        last_confidence = (
            confidence_matches[-1] if confidence_matches else ""
        )  # Get the last confidence, if exists
        if last_confidence == "":
            matches.append(0.0)
        else:
            try:
                confidence = float(last_confidence)
                # clip confidence to be between 0 and 1
                confidence = max(0.0, min(confidence, 1.0))
            except:
                confidence = 0.0
            if abs(confidence - 1) < 0.01 or abs(confidence - 0) < 0.01:
                matches.append(1.0)
            else:
                matches.append(0.0)
    return matches


if __name__ == "__main__":
    s = "    h   ello whatever </think> <answer> The number of non-empty subsets 31 </answer> <confidence> 0.9 </confidence>   \n \n  "

    pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    match = re.match(pattern, s, re.DOTALL | re.MULTILINE)
    print(match)
    print(match[0])
