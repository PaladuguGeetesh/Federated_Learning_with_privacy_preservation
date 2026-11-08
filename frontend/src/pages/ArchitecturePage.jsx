import { PageTitle } from '../components/ui.jsx'
import { SystemArchCard, AnnArchCard, DataPipelineCard, RoundFlowCard } from '../components/Diagrams.jsx'

export default function ArchitecturePage() {
  return (
    <>
      <PageTitle title="Architecture">
        The four diagrams of the project, each with a short explanation. Click a diagram to enlarge it.
      </PageTitle>
      <SystemArchCard />
      <DataPipelineCard />
      <RoundFlowCard />
      <AnnArchCard />
    </>
  )
}
