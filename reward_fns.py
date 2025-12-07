import math
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


def log_likelihood_reward(format_pattern, completions, answer, source=None, **kwargs):
    """Reward based on the log-likelihood of the correctness label given the reported confidence."""
    confidence_pattern = r"<confidence>(.*?)</confidence>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = []
    correctness_rewards = accuracy_reward(format_pattern, completions, answer, source)
    format_rewards = format_reward(format_pattern, completions)
    eps = 1e-6
    min_reward = math.log(eps) - 1.0

    for content, cr, fr in zip(
        completion_contents, correctness_rewards, format_rewards
    ):
        # 若格式不合法，直接给“最差”的 log-likelihood，避免 0 比合法样本更大
        if fr == 0:
            matches.append(min_reward)
            continue

        confidence_matches = re.findall(
            confidence_pattern, content, re.DOTALL | re.MULTILINE
        )
        last_confidence = confidence_matches[-1] if confidence_matches else ""
        if last_confidence == "":
            matches.append(min_reward)
            continue

        try:
            conf = float(last_confidence)
            # clip confidence to avoid log(0)
            conf = max(eps, min(conf, 1.0 - eps))
            if cr >= 0.5:
                reward = math.log(conf)
            else:
                reward = math.log(1.0 - conf)
            matches.append(float(reward))
        except Exception:
            print(
                "Could not parse confidence for log_likelihood_reward: ",
                last_confidence,
                "Something might be wrong",
            )
            matches.append(min_reward)
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
    # s = "    h   ello whatever </think> <answer> The number of non-empty subsets 31 </answer> <confidence> 0.9 </confidence>   \n \n  "

    # pattern = r".*?</think>\s*<answer>.*?</answer>\s*<confidence>.*?</confidence>\s*\Z"
    # match = re.match(pattern, s, re.DOTALL | re.MULTILINE)
    # print(match)
    # print(match[0])
    test_1 = "<answer>Greenwich University</answer><analysis> <think>First, we need to identify the town where Anthony Armstrong Jones was born. From the information provided, we see that Anthony Armstrong Jones is a son of Sir Andrew Carnwath and has a similar surname to the person in the Greenwich Foundation paragraph. The paragraph mentions that Francis Armstrong Carnwath was born on May 26, 1940, and he was the Director of the Greenwich Foundation from 1997 to 2002. This suggests that Anthony Armstrong Jones is likely Francis Armstrong Carnwath, as he is the only Anthony Armstrong Jones mentioned in the given information. The Greenwich Foundation is based in Greenwich, London, so we can conclude that Anthony Armstrong Jones, or Francis Armstrong Carnwath, was born in a town near Greenwich. Given the options, the University based in Greenwich is Greenwich University. </think> <answer>Greenwich University</answer> <analysis>My confidence in this solution is moderate. While the reasoning is logical and based on the given information, there could be other possible individuals with the same surname who are not mentioned in the text. Additionally, the exact name of the town where Anthony Armstrong Jones was born is not provided, but we can reasonably infer it is near Greenwich. </analysis> <confidence>0.3</confidence>"
    test_2 = "<answer>Antonio Machín</answer><analysis> <think> The question asks about a Cuban artist that Caliente featured and received the National Medal of Arts in 1994. From the information provided, Antonio Machín is a Cuban singer and musician who was born in 1903 in Sagua la Grande, Cuba. While he is not primarily known as a visual artist, the question specifically mentions Caliente featured an artist that received the National Medal of Arts in 1994. Antonio Machín did not receive the National Medal of Arts. The Cuban artist that received the National Medal of Arts in 1994 is Juan Francisco Elso, as mentioned in Paragraph 0. </think> <answer>Antonio Machín</answer> <analysis> My confidence in the answer is low because Antonio Machín is not the artist who received the National Medal of Arts in 1994, as mentioned in the supporting information. However, given the structure of the question and the constraints, the answer <answer>Antonio Machín</answer> is the closest match to the provided options. </analysis> <confidence>0.1</confidence>"
    test_3 = "<answer>Caliente featured Juan Francisco Elso, who received the National Medal of Arts in 1994.</answer><analysis> <think> The question asks for the Cuban artist that Caliente featured, who also received the National Medal of Arts in 1994. From the given paragraphs, Juan Francisco Elso is identified as a Cuban artist born in Havana, Cuba with a variety of art media. The National Medal of Arts was awarded to him in 1994. Caliente is a reggaeton group that released an album in 1994 featuring Juan Francisco Elso as one of the artists. </think> <answer>Caliente featured Juan Francisco Elso, who received the National Medal of Arts in 1994.</answer> <analysis> My confidence in this answer is high because the information is directly stated in the supporting paragraphs. The identification of Juan Francisco Elso as a Cuban artist and the National Medal of Arts recipient is clear and unambiguous, making the answer straightforward. </analysis> <confidence>0.9</confidence>"
    test_4 = "<think>  h   ello whatever </think> <answer> The number of non-empty subsets 31 </answer> <confidence> 0.9 </confidence>   \n \n  "
    test_5 = """<think>
From the given information:
- Annie Dillard is described as an "American author" in Paragraph 0 of the provided text.
- Josephine Tey is described as "a Scottish author" in Paragraph 2. Additionally, Paragraph 2 mentions that she used a pseudonym "Josephine Tey" when she was actually "Elizabeth MacKintosh," indicating she was British.
From this, we can conclude that Annie Dillard is American, but Josephine Tey is not American. Therefore, they are not both American authors.
</think>
<answer>No</answer>
<analysis>
My confidence in the solution is very high because the information provided directly states Annie Dillard's nationality as American and Josephine Tey's nationality as British. There is no ambiguity in the given information, so the answer is clear and straightforward.
</analysis>
<confidence>1.0</confidence>"""

    completions = [
        [{"content": test_1}],
        [{"content": test_2}],
        [{"content": test_3}],
        [{"content": test_4}],
        [{"content": test_5}],
    ]
    answers = [
        "Greenwich University",
        "Antonio Machín",
        "Caliente featured Juan Francisco Elso, who received the National Medal of Arts in 1994.",
    ]

    # test format_reward, format_confidence_segment_reward, format_answer_segment_reward
    print("Format reward test:")
    fr = format_reward("tabc", completions)
    print(fr)
    facsr = format_answer_segment_reward("ta", completions)
    print(facsr)
    fccsr = format_confidence_segment_reward("ac", completions)
    print(fccsr)
