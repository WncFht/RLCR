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
