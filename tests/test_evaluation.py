"""Whole-unit metric resampling and equal-graph risk checks."""
import unittest
import numpy as np
from cross_morse.evaluation import score_observations, MetricEvidence, paired_metric_interval
from cross_morse.tasks import metric_score, task_spec
from cross_morse.statistics import nested_selection, admission
from scripts.summarize_roman_splits import split_summary


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = {"dataset":"synthetic", "benchmark_dataset":"PascalVOC-SP", "data":{"synthetic_classes":2}}
        self.labels = np.array([0,0,1,1,0,1])
        self.units = np.array([2,2,2,7,7,9])
        self.p = np.array([[.8,.2],[.6,.4],[.1,.9],[.7,.3],[.6,.4],[.2,.8]])
        self.q = np.array([[.6,.4],[.4,.6],[.2,.8],[.2,.8],[.6,.4],[.4,.6]])

    def test_graph_loss_uses_equal_graph_means(self):
        scores=score_observations(self.p,self.labels,self.units,self.config)
        losses=-np.log(self.p[np.arange(6),self.labels])
        means=np.array([losses[:3].mean(),losses[3:5].mean(),losses[5]])
        np.testing.assert_allclose(scores["loss"],means)
        np.testing.assert_array_equal(scores["counts"],[1,1,1])
        self.assertAlmostEqual(scores["node_ce"],means.mean())
        self.assertNotAlmostEqual(scores["node_ce"],losses.mean())
        self.assertIsNone(scores["accuracy"])

    def test_crossfitted_metric_uses_predictions_and_bootstrap_frequencies(self):
        evidence=MetricEvidence([self.p,self.q],self.labels,self.units,[2,7,9],[1,1,1],self.config)
        comparators=np.array([0,1,0])
        reference=np.concatenate([self.p[:3],self.q[3:5],self.p[5:]])
        weights=np.array([2,2,2,1,1,0])
        spec=task_spec(self.config)
        expected=[metric_score(p,self.labels,spec,sample_weight=weights)-metric_score(reference,self.labels,spec,sample_weight=weights) for p in (self.p,self.q)]
        np.testing.assert_allclose(evidence([0,1],comparators,[2,1,0]),expected)
        core_config={"dataset":"synthetic","benchmark_dataset":"ogbn-proteins","data":{"synthetic_classes":2}}
        y=np.column_stack([self.labels,1-self.labels]).astype(float)
        core=MetricEvidence([self.p,self.q],y,self.units,[2,7,9],[3,2,1],core_config)
        target=metric_score(self.q,y,task_spec(core_config),sample_weight=weights)-metric_score(reference,y,task_spec(core_config),sample_weight=weights)
        self.assertAlmostEqual(core([1],comparators,[6,2,0])[0],target)

    def test_fixed_metric_self_comparison_is_zero_and_reproducible(self):
        a=paired_metric_interval(self.p,self.p,self.labels,self.units,self.config,reps=20,seed=91)
        b=paired_metric_interval(self.p,self.p,self.labels,self.units,self.config,reps=20,seed=91)
        self.assertEqual(a,b)
        self.assertEqual((a["mean"],a["lower"],a["upper"]),(0.,0.,0.))

    def test_nonlinear_nested_selection_calls_observation_evaluator(self):
        n=12
        y=np.tile([0,1],n); units=np.repeat(np.arange(n),2)
        p=np.tile([[.8,.2],[.3,.7]],(n,1)); q=np.tile([[.7,.3],[.2,.8]],(n,1))
        loss=np.column_stack([np.linspace(1,1.1,n),np.linspace(.96,1.06,n)])
        candidates=[{"route":"nominal","is_open_loop":True,"complexity":0}, {"route":"radius","is_open_loop":False,"complexity":1}]
        evidence=MetricEvidence([p,q],y,units,np.arange(n),np.ones(n),self.config)
        result=nested_selection(loss,None,candidates,np.arange(n)%3,inner_fold_ids=np.arange(n)%2,reps=20,seed=2,metric_evaluator=evidence)
        self.assertEqual(result["metric_mean"],[0.,0.])
        self.assertIn("metric_evaluation",result)
        failed=dict(route="phase",is_open_loop=True,complexity=0,execution_failure=True,feasible=False)
        expanded=MetricEvidence([p,np.full_like(p,np.nan),q],y,units,np.arange(n),np.ones(n),self.config)
        changed=nested_selection(np.column_stack([loss[:,0],np.full(n,np.nan),loss[:,1]]),None,
            [candidates[0],failed,candidates[1]],np.arange(n)%3,inner_fold_ids=np.arange(n)%2,reps=20,seed=2,metric_evaluator=expanded)
        self.assertEqual(changed["metric_mean"],[0.,None,0.])
        self.assertEqual(changed["provisional_index"],2)

    def test_independent_graph_admission_uses_conditional_bound(self):
        result=admission(np.full(200,-.04),np.zeros(200),weights=np.ones(200),reps=20,independent_graphs=True)
        self.assertTrue(result["passed"])
        self.assertIn("conditional_graph_bound",result)
        with self.assertRaisesRegex(ValueError,"equal"):
            admission([-.04,-.04],[0.,0.],weights=[1,2],reps=20,independent_graphs=True)

    def test_column_multiclass_labels_do_not_broadcast_accuracy(self):
        config={"dataset":"synthetic","data":{"synthetic_classes":2}}
        p=np.array([[.9,.1],[.1,.9],[.8,.2],[.3,.7]])
        labels=np.array([[0],[1],[0],[1]])
        score=score_observations(p,labels,[0,0,1,1],config)
        np.testing.assert_array_equal(score["accuracy"],[1.,1.])
        self.assertEqual(score["metric"],1.)

    def test_fully_missing_graph_is_not_a_metric_resampling_unit(self):
        config={"dataset":"synthetic","benchmark_dataset":"Peptides-func","data":{"synthetic_classes":2}}
        p=np.array([[.9,.2],[.2,.8],[.8,.3],[.4,.6],[.7,.4]])
        y=np.array([[1,0],[0,1],[1,0],[0,1],[np.nan,np.nan]])
        units=np.arange(5)
        score=score_observations(p,y,units,config)
        np.testing.assert_array_equal(score["core_ids"],[0,1,2,3])
        evidence=MetricEvidence([p,p],y,units,score["core_ids"],score["counts"],config)
        np.testing.assert_array_equal(evidence([0,1],np.zeros(4,dtype=int),np.ones(4)),[0.,0.])
        interval=paired_metric_interval(p,p,y,units,config,reps=20,seed=3)
        self.assertEqual(interval["units"],4)
        self.assertEqual(interval["mean"],0.)

    def test_roman_summary_requires_ten_and_uses_split_pairing(self):
        values=np.arange(10,dtype=float)/100
        result=split_summary(values)
        self.assertEqual(result["splits"],10)
        self.assertAlmostEqual(result["mean"],.045)
        self.assertLess(result["lower"],result["mean"])
        self.assertGreater(result["upper"],result["mean"])
        with self.assertRaises(ValueError):
            split_summary(values[:9])


if __name__ == "__main__":
    unittest.main()
