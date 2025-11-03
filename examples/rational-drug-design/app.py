# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABNQSSSimulator:
    """ABN-QSS 模拟器核心"""
    
    def __init__(self, network_size: int = 4):
        self.n = network_size
        # 定义简化生理网络： [食欲, 能耗, 血糖, 胃肠道副作用]
        self.node_names = ['Appetite', 'Energy Expenditure', 'Blood Sugar', 'GI Side Effect']
        # 初始化魔方阵约束的连接矩阵 (目标是抑制食欲、促进能耗、控制血糖、降低副作用)
        self.magic_matrix = np.array([
            [0, 0.5, 0.3, -0.8],   # 食欲中枢受能耗、血糖正向调节，但受GI副作用负向调节（恶心降低食欲）
            [0.7, 0, -0.2, -0.5],  # 能耗受食欲和血糖刺激，但受GI副作用抑制
            [0.3, -0.1, 0, 0.2],   # 血糖受食欲正向调节，被能耗轻微负向调节
            [-0.6, -0.3, 0.1, 0]   # GI副作用被食欲和能耗强烈抑制（理想药物特征），被血糖轻微激发
        ])
        
    def simulate(self, molecular_profile: List[float], max_iter: int = 500) -> Dict[str, Any]:
        """
        模拟分子在生理网络中的效应
        molecular_profile: 分子对4个节点的直接作用强度 [ΔAppetite, ΔEnergy, ΔBloodSugar, ΔGI]
        """
        logger.info(f"开始模拟，分子作用谱: {molecular_profile}")
        V = np.array([0.0, 0.0, 0.0, 0.0])  # 初始状态
        noise_level = 0.02  # 模拟生物噪声
        convergence_threshold = 1e-4
        history = []
        
        for i in range(max_iter):
            # 系统动力学: dV/dt = -V + (W * V) + I + noise
            I = np.array(molecular_profile)  # 分子直接输入
            dv = -V + self.magic_matrix @ V + I
            V_next = V + 0.1 * dv + np.random.normal(0, noise_level, self.n)
            
            history.append(V_next.copy())
            if np.linalg.norm(V_next - V) < convergence_threshold:
                logger.info(f"系统在第 {i} 次迭代后收敛")
                break
            V = V_next
        else:
            logger.warning(f"系统在 {max_iter} 次迭代后未完全收敛")
            
        steady_state = V
        # 计算综合评分：效益(食欲抑制+能耗提升) - 风险(副作用)
        benefit = -steady_state[0] + steady_state[1]  # 抑制食欲为负，转为正；提升能耗为正
        risk = steady_state[3]  # GI副作用
        composite_score = benefit - risk
        
        return {
            'steady_state': steady_state,
            'history': np.array(history),
            'composite_score': composite_score,
            'benefit': benefit,
            'risk': risk
        }

class LLMDesigner:
    """LLM 驱动的分子设计师 (简化模拟版本)"""
    
    def __init__(self):
        # 模拟一个知识库：预先定义好的候选分子及其作用谱
        self.molecule_library = {
            "Molecule_A (GLP-1优化)": [-0.9, 0.3, -0.8, -0.4],  # 强效降糖抑食，中等能耗，低副作用
            "Molecule_B (双靶点激动)": [-0.7, 0.9, -0.6, -0.7],  # 强效能耗提升，强副作用抑制
            "Molecule_C (平衡型)": [-0.8, 0.6, -0.7, -0.2],     # 各方面均衡
            "Molecule_D (高风险高回报)": [-0.95, 0.8, -0.5, 0.3], # 极强效，但副作用也高
        }
        
    def generate_candidates(self, design_goal: str, n_candidates: int = 3) -> List[Dict[str, Any]]:
        """根据设计目标生成候选分子 (模拟LLM推理)"""
        logger.info(f"LLM接收设计目标: {design_goal}")
        
        # 模拟LLM根据设计目标筛选和生成分子
        candidates = []
        for name, profile in self.molecule_library.items():
            score = self._evaluate_design_match(profile, design_goal)
            candidates.append({
                'name': name,
                'molecular_profile': profile,
                'rationale': self._generate_rationale(name, profile, design_goal),
                'match_score': score
            })
        
        # 按匹配度排序并返回前n个
        candidates.sort(key=lambda x: x['match_score'], reverse=True)
        return candidates[:n_candidates]
    
    def _evaluate_design_match(self, profile: List[float], goal: str) -> float:
        """评估分子与设计目标的匹配度"""
        goal = goal.lower()
        score = 0.0
        
        # 简化的规则：解析设计目标关键词并评分
        if '能耗' in goal or '能量消耗' in goal or '生热' in goal:
            score += profile[1] * 2.0  # 能耗权重高
        if '副作用' in goal or '胃肠道' in goal or 'gi' in goal:
            score -= abs(profile[3]) * 1.5  # 副作用低则加分
        if '食欲' in goal or '抑制食欲' in goal:
            score -= profile[0] * 1.2  # 食欲降低为负值，取反后为正向加分
        if '血糖' in goal or '降糖' in goal:
            score -= profile[2] * 1.0  # 血糖降低为负值，取反后为正向加分
            
        return score
    
    def _generate_rationale(self, name: str, profile: List[float], goal: str) -> str:
        """生成设计原理说明 (模拟LLM的推理输出)"""
        effects = []
        if profile[0] < -0.7:
            effects.append("强效抑制食欲")
        elif profile[0] < -0.4:
            effects.append("中等抑制食欲")
            
        if profile[1] > 0.7:
            effects.append("显著提升能量消耗")
        elif profile[1] > 0.4:
            effects.append("适度提升能量消耗")
            
        if profile[3] < -0.6:
            effects.append("极低胃肠道副作用风险")
        elif profile[3] < -0.3:
            effects.append("较低胃肠道副作用风险")
        elif profile[3] > 0.2:
            effects.append("⚠️ 注意：存在胃肠道副作用风险")
        
        return f"{name} 通过多靶点调节，实现{'+'.join(effects)}。该设计专门针对'{goal}'进行优化。"

def main():
    from 荒謬絕倫的製藥研究 import render_exposé
    page = st.sidebar.selectbox("选择页面", ["理性设计模拟器", "药物研发现形记"])
    if page == "理性设计模拟器":
        st.set_page_config(page_title="理性药物设计模拟器 v0.1", layout="wide")
        st.title("🧬 ABN-QSS × LLM 理性药物设计模拟器")
        st.markdown("**从‘盲盒发现’到‘系统设计’—— 新一代药物研发范式验证**")
        
        # 初始化模拟器
        if 'simulator' not in st.session_state:
            st.session_state.simulator = ABNQSSSimulator()
            st.session_state.designer = LLMDesigner()
        
        # 用户输入区域
        st.header("1. 设定您的设计目标")
        design_goal = st.text_area(
            "用自然语言描述您理想的药物特性:",
            value="设计一个在强效抑制食欲和提升能量消耗的同时，能最大限度降低胃肠道副作用的分子",
            height=80
        )
        
        if st.button("🚀 生成并评估候选分子", type="primary"):
            with st.spinner("LLM正在生成候选分子，ABN-QSS网络正在进行系统级模拟评估..."):
                # 步骤1: LLM生成候选分子
                candidates = st.session_state.designer.generate_candidates(design_goal)
                
                # 步骤2: 对每个候选分子进行模拟
                results = []
                for candidate in candidates:
                    sim_result = st.session_state.simulator.simulate(candidate['molecular_profile'])
                    candidate.update(sim_result)
                    results.append(candidate)
                
                st.session_state.results = results
        
        # 显示结果
        if 'results' in st.session_state:
            st.header("2. 候选分子系统评估报告")
            
            # 排序结果
            results = sorted(st.session_state.results, key=lambda x: x['composite_score'], reverse=True)
            best_candidate = results[0]
            
            # 显示最佳分子
            st.subheader(f"🏆 最佳候选: {best_candidate['name']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("综合评分", f"{best_candidate['composite_score']:.2f}")
            with col2:
                st.metric("效益指数", f"{best_candidate['benefit']:.2f}")
            with col3:
                st.metric("风险指数", f"{best_candidate['risk']:.2f}")
            
            st.write(f"**设计原理:** {best_candidate['rationale']}")
            
            # 可视化：稳态效果雷达图
            st.subheader("📊 系统效应可视化")
            fig = go.Figure()
            
            for res in results:
                fig.add_trace(go.Scatterpolar(
                    r=res['steady_state'],
                    theta=st.session_state.simulator.node_names,
                    fill='toself',
                    name=res['name'],
                    opacity=0.7
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 详细结果表格
            st.subheader("📋 详细模拟数据")
            display_data = []
            for res in results:
                display_data.append({
                    '候选分子': res['name'],
                    '综合评分': f"{res['composite_score']:.2f}",
                    '食欲抑制': f"{-res['steady_state'][0]:.2f}",
                    '能耗提升': f"{res['steady_state'][1]:.2f}",
                    '血糖控制': f"{-res['steady_state'][2]:.2f}",
                    'GI副作用': f"{res['steady_state'][3]:.2f}",
                })
            
            st.table(display_data)
            
            # 范式转变的总结
            st.header("3. 范式转变总结")
            st.success("""
            **从‘威而刚式’的偶然发现，走向‘理性设计’的必然之路：**
            - ✅ **系统性**：同时评估疗效与安全性，而非事后发现副作用
            - ✅ **预见性**：在合成前预测分子在复杂生理网络中的行为
            - ✅ **可解释性**：提供作用机制的物理解释，告别黑箱
            - ✅ **高效性**：将传统数月甚至数年的初期筛选缩短至分钟级
            """)
    else:
            render_exposé()

if __name__ == "__main__":
        main()
