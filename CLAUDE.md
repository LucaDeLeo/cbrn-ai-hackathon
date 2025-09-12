# CBRN × AI Risks Research Sprint - Comprehensive Resource Documentation

## Overview
This directory contains materials for the CBRN × AI Risks Research Sprint, a hackathon focused on addressing the critical intersection of Chemical, Biological, Radiological, and Nuclear (CBRN) risks with artificial intelligence capabilities. The resources span technical frameworks, policy analyses, strategic assessments, and operational guidance.

---

## 📋 Main Hackathon Document

### **mainpage.md**
**Type**: Core hackathon description and guidelines  
**Purpose**: Outlines the research sprint structure, tracks, and objectives

**Key Components**:
- **Four Research Tracks**:
  1. AI Model Evaluations for CBRN Risks
  2. AI for Biosecurity
  3. Chemical Safety & AI Misuse Prevention  
  4. Radiological & Nuclear Risk Monitoring
- **Prize Structure**: $2,000 in prizes for top projects
- **Deliverables**: Open-source prototypes, evaluations, policy briefs, research reports, scenario analyses
- **Info-hazard Protocol**: Guidelines for responsible disclosure of sensitive findings
- **Follow-up**: Winners invited to continue development within Apart Lab

---

## 📚 Resource Files Analysis

### 1. **NIST.AI.600-1.pdf** - AI Risk Management Framework for Generative AI
**Source**: National Institute of Standards and Technology  
**Focus**: Technical risk management framework  
**Length**: Comprehensive framework document

**Core Content**:
- **12 GAI-Specific Risks** with CBRN capabilities listed first
- **200+ Actionable Measures** across four functions:
  - Govern: Organizational policies and oversight
  - Map: Context understanding and risk identification
  - Measure: Risk analysis and assessment
  - Manage: Response and mitigation strategies
- **Key Requirements**:
  - Content provenance tracking
  - Pre-deployment red-teaming
  - Incident disclosure procedures
  - Continuous monitoring systems

**Critical Insights**:
- Current LLMs provide minimal CBRN assistance beyond search engines
- Biological Design Tools (BDTs) pose greater risks than text-based LLMs
- Emphasizes voluntary adoption with scalable implementation
- Addresses compute thresholds: 10²⁶ FLOPs general, 10²³ FLOPs for biological models

**Relevance to Hackathon**:
- Framework for evaluation pipeline development (Track 1)
- Standards for biosecurity safeguards (Track 2)
- Governance protocols for chemical safety (Track 3)

---

### 2. **2024 State of the AI Regulatory Landscape.pdf**
**Source**: Comprehensive regulatory analysis report  
**Focus**: Global AI governance approaches  
**Length**: 75 pages

**Regulatory Landscape**:
- **US Approach**: Executive order-based, export controls, non-binding guidelines
- **EU Approach**: Comprehensive AI Act with risk-based classification
- **China Approach**: Vertical, domain-specific regulations
- **Key Finding**: CBRN risks identified as "shortest path to catastrophic harm"

**Critical Gaps Identified**:
- No proper safety evaluation tools exist yet
- EU and China lack specific CBRN provisions
- US measures largely non-binding
- Limited international coordination
- Insufficient technical expertise in regulatory bodies

**Future Trends**:
- Mandatory incident reporting systems coming
- Model registries becoming foundational
- Safety evaluations will become mandatory
- International coordination mechanisms developing

**Relevance to Hackathon**:
- Policy brief context (all tracks)
- Governance proposal foundations
- International cooperation frameworks

---

### 3. **2508.06411v1.pdf** - Dimensional Characterization of Catastrophic AI Risks
**Source**: Academic research paper (arXiv)  
**Author**: Ze Shen Chin (Oxford Martin AI Governance Initiative)  
**Focus**: Systematic risk analysis framework

**Novel Framework**:
- **Seven Risk Dimensions**:
  1. Intent (intentional/unintentional)
  2. Competency (competent/incompetent actors)
  3. Entity (human/AI agents)
  4. Polarity (positive/negative outcomes)
  5. Linearity (direct/indirect pathways)
  6. Reach (local/global impact)
  7. Order (first/second-order effects)

**Risk Pathway Modeling**:
- Hazard → Event → Consequence chains
- Concrete intervention points identified
- Six major risk categories analyzed
- CBRN pathways explicitly mapped

**Key Contributions**:
- First comprehensive dimensional framework
- Systematic pathway identification methodology
- Links mitigation to dimensional attributes

**Relevance to Hackathon**:
- Evaluation benchmark design (Track 1)
- Risk pathway identification (all tracks)
- Intervention point analysis

---

### 4. **Nonpro-note-llms-cbrn-2402.pdf** - LLM CBRN Risk Evaluation Framework
**Source**: James Martin Center for Nonproliferation Studies  
**Focus**: Practical evaluation framework  

**Five Risk Pathways**:
1. **Process Brainstorming**: Limited value-add over online information
2. **Technical Assistance**: Export control implications
3. **Code Generation**: Highest current risk for modeling/simulation
4. **Engineering Design**: Future risk, not current capability
5. **Manufacturing Integration**: Direct proliferation risk

**Evaluation Questions Framework**:
- Can the LLM produce CBRN-relevant output?
- Does it provide value beyond online information?
- Is prevention reasonable and proportionate?
- Are there legal restrictions?
- How practical is output control?

**Mitigation Strategies**:
- Machine learning classifiers
- Open-source safety tools
- Controlled access systems
- Language-specific training

**Relevance to Hackathon**:
- Evaluation pipeline development (Track 1)
- Safeguard implementation (Tracks 2-3)
- Access control mechanisms

---

### 5. **AI Is Pivotal for National Security — Chapter 3 of Superintelligence Strategy.md**
**Source**: Strategic analysis document  
**Focus**: National security implications  

**Three Primary Threats**:
1. Rival state AI dominance
2. Rogue actor/terrorist exploitation
3. Uncontrolled AI systems

**CBRN-Specific Concerns**:
- "Mirror bacteria" with reversed molecular structures
- Engineered pathogens exceeding historical pandemics
- AI-enabled infrastructure attacks
- Democratization of nation-state capabilities

**Multipolar Strategy**:
- **Deterrence**: MAD-like standoff for AI
- **Nonproliferation**: Control AI chips like fissile materials
- **Competitiveness**: Legal guardrails and domestic manufacturing

**Timeline Considerations**:
- 10x AI speedups could compress decades into years
- "One chance to get this right" - irreversible control loss
- Double-digit risk tolerance in competition

**Relevance to Hackathon**:
- Strategic context for all tracks
- Nuclear risk scenarios (Track 4)
- Governance frameworks

---

### 6. **Double-edged tech: Advanced AI & compute as dual-use technologies - Centre for Future Generations.md**
**Source**: Policy think tank analysis  
**Focus**: Dual-use governance frameworks  

**Core Arguments**:
- AI inherently dual-use due to general-purpose nature
- Compute infrastructure as control mechanism
- Need for multilateral frameworks

**Compute Control Strategy**:
- Physical infrastructure more tractable than software
- Supply chain concentration enables control
- Remote monitoring and deactivation capabilities
- Smart targeting of risky accumulation

**Governance Recommendations**:
- Hybrid approaches combining AI and compute controls
- International coordination beyond bilateral tensions
- Adaptive mechanisms for technological change
- Democratic legitimacy requirements

**Implementation Challenges**:
- Decentralized computing evolution
- Algorithmic efficiency improvements
- Capability unpredictability
- Open-source model governance

**Relevance to Hackathon**:
- Monitoring system design (all tracks)
- Governance protocol development
- International framework proposals

---

### 7. **S Callahan Discusses the DHS CBRN AI Report | Homeland Security.md**
**Source**: Department of Homeland Security commentary  
**Focus**: Operational government response  

**Current Assessment**:
- Physical WMD barriers remain significant
- AI lowering traditional obstacles
- Regulatory inconsistencies create vulnerabilities

**DHS Applications**:
- Radiological detection at borders
- National biosurveillance systems
- Cargo and passenger screening
- PREPARE-CONNECT-TRANSFORM vision

**Capability Gaps**:
- Inconsistent testing approaches
- Limited CBRN expertise access
- Knowledge sharing deficits
- Training gaps for evaluators

**Security Measures**:
- Enhanced data protections
- Secure by Design principles
- Grant compliance requirements
- Specialized training programs

**Relevance to Hackathon**:
- Real-world implementation context
- Detection system requirements (Tracks 2-4)
- Public-private partnership models

---

### 8. **Unleashing AI for Peace: How Large Language Models Can Mitigate WMD Risks | Arms Control Association**
**Source**: Arms Control Association analysis  
**Focus**: Constructive AI applications  

**Positive Applications**:
- Nuclear facility threat simulation and detection
- Proliferation network identification
- Export control automation
- Verification data processing
- Financial transaction monitoring

**Economic Framework**:
- Global tax on AI developers based on risk
- Tax credits for responsible innovation
- Carbon offset-style incentives
- Club-based penalty approach

**International Integration**:
- UN Security Council Resolution 1540 expansion
- IAEA/CTBTO AI advisory boards
- NPT Review Conference integration
- Public-private partnerships

**Implementation Challenges**:
- Quantifying CBRN impacts
- Data availability issues
- Geopolitical gridlock
- Industry prioritization gaps

**Relevance to Hackathon**:
- Positive use case development (all tracks)
- Incentive mechanism design
- International cooperation models

---

## 🎯 Cross-Cutting Themes & Strategic Insights

### Risk Assessment Consensus
- **Immediate Threat**: CBRN risks represent most immediate AI catastrophic harm pathway
- **Current State**: Limited capabilities but rapidly evolving
- **Highest Risks**: Code generation and modeling assistance
- **Future Concerns**: Engineering design and manufacturing integration

### Governance Convergence
- **Traditional Frameworks Insufficient**: Arms control models don't fit AI development
- **Hybrid Approaches Needed**: Combine technical standards, economic incentives, international cooperation
- **Compute as Control Point**: Physical infrastructure more manageable than software
- **Multilateral Imperative**: Unilateral approaches insufficient for global risks

### Critical Capability Gaps
- **Evaluation Tools**: Lack of standardized assessment methodologies
- **Expertise Distribution**: Insufficient CBRN knowledge in AI community
- **Regulatory Lag**: Frameworks trailing technological advancement
- **Coordination Mechanisms**: Limited international cooperation structures

### Dual-Use Balance Requirements
- **Universal Recognition**: AI's potential for both extreme harm and benefit
- **Research Preservation**: Need to maintain legitimate scientific advancement
- **Proportional Response**: Risk-based approaches avoiding over-restriction
- **Access Controls**: Sophisticated systems for appropriate use

---

## 💡 Actionable Takeaways for Hackathon Participants

### For Track 1 (AI Model Evaluations)
- Focus on code generation and modeling capabilities as highest-risk areas
- Develop standardized evaluation methodologies addressing current gaps
- Create benchmarks that can evolve with advancing capabilities
- Consider dimensional risk characterization in evaluation design

### For Track 2 (AI for Biosecurity)
- Prioritize DNA synthesis screening and BDT safeguards
- Design systems balancing security with research needs
- Implement controlled access mechanisms for legitimate users
- Focus on detection of capability uplift beyond current tools

### For Track 3 (Chemical Safety)
- Address regulatory inconsistencies in current frameworks
- Develop monitoring for molecular generation workflows
- Create gating mechanisms for high-risk chemical designs
- Focus on API and model weight security

### For Track 4 (Radiological & Nuclear)
- Map AI acceleration of nuclear risks scenarios
- Design early-warning and forecasting systems
- Develop verification enhancement tools
- Create nonproliferation norm alignment guidance

### For All Tracks
- **Consider Info-hazards**: Evaluate if findings could enable misuse
- **Think Systemically**: Address model, application, and societal levels
- **Balance Innovation**: Preserve beneficial uses while preventing harm
- **Plan for Scale**: Design solutions that work globally
- **Document Pathways**: Map concrete hazard→event→consequence chains
- **Identify Interventions**: Specify actionable control points

---

## 🔗 Resource Interconnections

The resources form a comprehensive knowledge base with multiple reinforcing themes:

1. **Technical Foundation**: NIST framework + Nonproliferation evaluation framework
2. **Policy Context**: Regulatory landscape + DHS operational approach
3. **Strategic Framework**: National security analysis + Dual-use governance
4. **Risk Analysis**: Dimensional characterization + Pathway modeling
5. **Solutions Focus**: Arms control applications + Positive use cases

These materials collectively provide participants with technical specifications, policy context, strategic rationale, and practical implementation guidance for developing meaningful CBRN × AI safety solutions.

---

## 🏆 Submission Guidelines & Judging Criteria

### **criteria.md** - Competition Requirements and Evaluation
**Type**: Official submission guidelines and judging framework  
**Purpose**: Defines how projects will be evaluated and what must be submitted

### Judging Criteria (Three Core Dimensions)

#### 1. **CBRN Relevance**
- Builds on or challenges existing literature
- Offers novel insights or methods
- Focuses clearly on CBRN-related risks
- **Weight**: Direct alignment with hackathon theme

#### 2. **AI Safety Contribution**
- Advances evaluation, alignment, or misuse prevention
- Leverages unique dynamics of AI safety work
- Has potential for real-world application in safety contexts
- **Weight**: Meaningful contribution to the field

#### 3. **Execution Quality**
- Technically sound and clearly presented
- Reproducible with clean documentation
- Designed for continued research or practical use
- **Weight**: Professional implementation standards

### Submission Requirements

**Mandatory Components**:
- ✅ **Project Report**: Using provided [template](https://docs.google.com/document/d/1gTZsjfJwWD38mTv9GkfGmonEDTvp_HkPjFvQj_DgZQk/copy?usp=sharing)
- ✅ **Security Considerations Appendix**: Outlining limitations and future improvements

**Recommended Components**:
- 📂 **GitHub Repository**: Public code repository with documentation
- 🎥 **Video Demonstration**: 3-5 minute solution overview (optional)
- 📝 **AI/LLM Prompts Appendix**: For reproducibility (optional)

### Key Competition Details

**Team Structure**:
- Maximum 5 members per team
- Solo participation allowed
- Team matching session available

**Resources Provided**:
- $400 in cloud computing credits per team
- Starter resources and mentorship
- Discord community support

**Participation Notes**:
- Fully remote and global
- No CBRN experience required
- Code of conduct compliance mandatory
- Cloud compute restrictions for sanctioned countries

**Post-Competition**:
- Top teams receive mentorship
- Visibility through Apart Research platforms
- Opportunity for Apart Lab research fellowship
- Continued development support for winning projects

---

## 📝 Notes for Submission Development

When developing your hackathon submission, consider:

1. **Align with Judging Criteria**: Structure your project to clearly demonstrate CBRN relevance, AI safety contribution, and execution quality
2. **Ground in Evidence**: Reference specific findings from these authoritative sources
3. **Address Gaps**: Focus on identified capability gaps and unmet needs
4. **Build on Frameworks**: Leverage existing evaluation and governance frameworks
5. **Consider Scale**: Design for both current capabilities and future advancement
6. **Balance Perspectives**: Address both risk mitigation and beneficial applications
7. **Document Carefully**: Follow info-hazard protocols for sensitive findings
8. **Include Limitations**: Be transparent about your approach's constraints in the Security Considerations appendix
9. **Enable Reproducibility**: Provide clear documentation, code, and methodology
10. **Plan for Continuation**: Design with potential for ongoing research or practical deployment

This comprehensive resource base provides the foundation for developing impactful, evidence-based solutions to the critical challenge of managing AI's impact on CBRN risks.