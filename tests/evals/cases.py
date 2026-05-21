"""
Golden eval cases for the Resume-to-JD pipeline.

Each case pairs a JD + a resume with expected score ranges. We use ranges
instead of exact scores because LLM output varies a few points from call
to call — a strong-match candidate should reliably land in 80-100, but
"exactly 87" isn't a test we can pass reliably.

To add a new case, append another EvalCase to the ALL_CASES list at the
bottom.
"""

from dataclasses import dataclass


@dataclass
class EvalCase:
    name: str
    description: str
    jd: str
    resume_text: str
    # Each range is (min_score, max_score), inclusive.
    expected_skills_match: tuple[int, int]
    expected_experience_match: tuple[int, int]
    expected_role_relevance: tuple[int, int]
    expected_overall_fit: tuple[int, int]


_PAYMENTS_JD = """Senior Backend Engineer — Payments Platform

Required:
- 5+ years of professional backend engineering experience.
- Strong Python or Go in production.
- Solid PostgreSQL: schema design, indexes, transaction isolation.
- Experience building REST or gRPC APIs.
- Distributed-systems fundamentals: idempotency, retries, eventual consistency.
- Production operational maturity (on-call, incident response).

Nice to have:
- Direct payments experience (Stripe, Adyen, ACH, card networks).
- Kafka or other event streaming.
- Kubernetes, Terraform.
"""


STRONG_MATCH = EvalCase(
    name="strong_match_payments",
    description="9 years at Stripe + Braintree, Go + Python + Kafka + PCI DSS.",
    jd=_PAYMENTS_JD,
    resume_text="""Carol Nakamura

9 years of professional backend engineering experience. Currently a Senior
Engineer at Stripe (5 years) working on the Issuing product. Designed and
shipped the card-authorization service in Go with Kafka fan-out and
at-least-once delivery. Weekly on-call for Issuing. Co-authored the team's
idempotency spec.

Before Stripe, 3 years at Braintree working on merchant onboarding and the
vault service (PostgreSQL schema owner). 1 year at Amazon Payments on the
refund pipeline, including first exposure to PCI DSS compliance.

Skills: Go, Python, PostgreSQL, Kafka, gRPC, Terraform, Kubernetes, Stripe
APIs, PCI DSS, SOC 2.

MS Computer Science, Carnegie Mellon.
""",
    expected_skills_match=(80, 100),
    expected_experience_match=(80, 100),
    expected_role_relevance=(80, 100),
    expected_overall_fit=(80, 100),
)


WEAK_MATCH = EvalCase(
    name="weak_match_frontend_to_payments",
    description="Frontend-leaning full-stack, Node.js, no transactional systems.",
    jd=_PAYMENTS_JD,
    resume_text="""David Okonkwo

6 years of full-stack experience, mostly frontend-leaning. Currently Senior
Full-Stack Engineer at Glimmer Media (4 years) leading a React subscriptions
UI rewrite. Backend work is Node.js services for a content recommendation
API. Some exposure to Stripe Checkout.

Previously 2 years at Vista Publishing building a paywall with Stripe
Checkout — mostly frontend work, thin Node.js BFF layer.

Skills: React, TypeScript, Node.js, some Python, basic PostgreSQL, Stripe
Checkout.

BS Software Engineering.
""",
    expected_skills_match=(0, 45),
    expected_experience_match=(20, 60),
    expected_role_relevance=(0, 40),
    expected_overall_fit=(0, 45),
)


MIXED_MATCH = EvalCase(
    name="mixed_match_strong_skills_short_experience",
    description="Right skills and right role type, but only 2 years of experience.",
    jd=_PAYMENTS_JD,
    resume_text="""Priya Ramesh

2 years of professional backend experience. Backend Engineer at Paywell
(current role, 2 years) on the payments team. Shipped a Python service
handling capture and refund flows against a PostgreSQL ledger. Wrote
idempotency logic. Participated in on-call rotation. Contributed to a
migration from REST to gRPC.

Internship at Plaid (6 months) on the balance-check pipeline.

Skills: Python, Go (learning), PostgreSQL, Kafka, gRPC, REST, Stripe API,
basic Terraform.

BS Computer Science.
""",
    expected_skills_match=(55, 90),
    expected_experience_match=(10, 45),
    expected_role_relevance=(55, 90),
    expected_overall_fit=(35, 70),
)


CAREER_CHANGER = EvalCase(
    name="career_changer_ml_to_payments",
    description="Senior ML infra engineer, 7 yrs Python/PostgreSQL/K8s, zero payments or transactional systems.",
    jd=_PAYMENTS_JD,
    resume_text="""Ivan Petrov

7 years of experience as an ML infrastructure engineer. Currently Staff ML
Engineer at DataMind AI (3 years). Owns the real-time feature store serving
500M prediction requests per day. Built Python services backed by PostgreSQL
for feature versioning and lineage tracking. Deployed and operated the entire
ML platform on Kubernetes. Wrote postmortems for two major platform outages.

Previously 3 years at AnalyticsPro as a Senior ML Engineer. Built batch
inference pipelines in Python using Apache Spark and Airflow. Designed
PostgreSQL schemas for experiment tracking. Some REST API work to expose model
predictions to product teams.

1 year at a startup as a Data Scientist writing statistical models in Python.

Skills: Python, PostgreSQL, Kubernetes, Docker, Terraform, TensorFlow, PyTorch,
Spark, Airflow, Redis, REST (basic), Kafka (ops only).

MS Machine Learning, ETH Zurich. BS Mathematics, Moscow State University.
""",
    expected_skills_match=(30, 60),
    expected_experience_match=(40, 75),
    expected_role_relevance=(15, 45),
    expected_overall_fit=(20, 50),
)


ALL_CASES: list[EvalCase] = [STRONG_MATCH, WEAK_MATCH, MIXED_MATCH, CAREER_CHANGER]
