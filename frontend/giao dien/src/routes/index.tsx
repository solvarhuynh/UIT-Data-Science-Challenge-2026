import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { Bookmark, BookmarkX, Menu } from "lucide-react";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { PersistentSidebar, Sidebar, type WorkspaceView } from "@/components/legal/Sidebar";
import { TopBar } from "@/components/legal/TopBar";
import { WelcomeState } from "@/components/legal/WelcomeState";
import { PromptComposer } from "@/components/legal/PromptComposer";
import { AssistantAnswer, UserMessage } from "@/components/legal/Messages";
import { LoadingPipeline } from "@/components/legal/LoadingPipeline";
import { EvidencePanelContent, type EvidenceTab } from "@/components/legal/EvidencePanel";
import { LegalSearch } from "@/components/legal/LegalSearch";
import { WarningBanner } from "@/components/legal/WarningBanner";
import { SettingsDialog } from "@/components/legal/SettingsDialog";
import { buildRequest, runLegalQuery, suggestFollowUps, type PipelineStage } from "@/lib/legal/api";
import { buildSeedConversations, type LegalDocEntry } from "@/lib/legal/mock-data";
import { savedSourceFromCitation, savedSourceFromDoc } from "@/lib/legal/saved";
import { usePersistentState, useTheme } from "@/lib/legal/storage";
import type {
  Citation,
  Conversation,
  Message,
  QueryMode,
  QueryRequest,
  QueryResponse,
  RetrievalHit,
  RetrievalSettings,
  SavedLegalSource,
} from "@/lib/legal/types";

export const Route = createFileRoute("/")({
  component: Workspace,
  head: () => ({
    meta: [
      { title: "HCMUTE-SHIPCODE — Trợ lý pháp luật Việt Nam" },
      {
        name: "description",
        content:
          "Hệ thống tra cứu và hỏi đáp pháp luật Việt Nam: mọi câu trả lời đều gắn với điều, khoản và căn cứ pháp lý được truy hồi.",
      },
      { property: "og:title", content: "HCMUTE-SHIPCODE — Trợ lý pháp luật Việt Nam" },
      {
        property: "og:description",
        content: "Hệ thống tra cứu và hỏi đáp pháp luật dựa trên căn cứ pháp lý được truy hồi.",
      },
      { property: "og:type", content: "website" },
      { property: "og:url", content: "/" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
    links: [{ rel: "canonical", href: "/" }],
  }),
});

const DEFAULT_SETTINGS: RetrievalSettings = {
  topK: 10,
  lawName: "",
  docId: "",
  article: "",
  advanced: false,
};

function uid() {
  return `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e6).toString(36)}`;
}

function hasConversationContent(conversation: Conversation) {
  return (
    Array.isArray(conversation.messages) &&
    conversation.messages.some(
      (message) => typeof message.content === "string" && message.content.trim().length > 0,
    )
  );
}

interface RetryPayload {
  question: string;
  conversationId: string;
  messageId: string;
  mode: QueryMode;
  request: QueryRequest;
}

function evidenceFromSavedSource(source: SavedLegalSource) {
  const citation: Citation = {
    id: source.id,
    label: "Nguồn đã chọn",
    title: source.title,
    doc_id: source.doc_id,
    law_name: source.law_name,
    doc_number: source.doc_number,
    article: source.article,
    clause: source.clause,
    point: source.point,
    excerpt: source.excerpt,
    chunk_id: `selected:${source.id}`,
    score: 1,
    source: source.source ?? "Kho văn bản HCMUTE-SHIPCODE",
    rank: 1,
  };

  const hit: RetrievalHit = {
    chunk_id: citation.chunk_id,
    doc_id: source.doc_id ?? source.doc_number ?? source.law_name,
    text: source.excerpt,
    score: 1,
    source: citation.source,
    law_name: source.law_name,
    article: source.article,
    clause: source.clause,
    point: source.point,
    dense_score: 1,
    sparse_score: 1,
    hybrid_score: 1,
    rerank_score: 1,
    final_score: 1,
    rank: 1,
    metadata: {
      doc_id: source.doc_id ?? source.doc_number ?? source.law_name,
      law_name: source.law_name,
      doc_number: source.doc_number,
      article: source.article,
      clause: source.clause,
      point: source.point,
      source: source.source,
    },
  };

  return { citation, hit };
}

function Workspace() {
  const { theme, toggle: toggleTheme } = useTheme();
  const [conversations, setConversations, conversationsHydrated] = usePersistentState<
    Conversation[]
  >("shipcode.conversations", buildSeedConversations());
  const [savedSources, setSavedSources] = usePersistentState<SavedLegalSource[]>(
    "shipcode.saved-sources",
    [],
  );
  // Mỗi lần mở ứng dụng đều bắt đầu tại trang tra cứu rỗng; lịch sử vẫn được giữ lại.
  const [activeId, setActiveId] = useState<string | null>(null);

  const [view, setView] = useState<WorkspaceView>("qa");
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<QueryMode>("qa");
  const [settings, setSettings] = useState<RetrievalSettings>(DEFAULT_SETTINGS);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [activeCitation, setActiveCitation] = useState<string | null>(null);
  const [evidenceMessageId, setEvidenceMessageId] = useState<string | null>(null);
  const [evidenceTab, setEvidenceTab] = useState<EvidenceTab>("citations");
  const [selectedSource, setSelectedSource] = useState<SavedLegalSource | null>(null);

  const [stage, setStage] = useState<PipelineStage>("done");
  const [liveHits, setLiveHits] = useState<RetrievalHit[]>([]);
  const [streamText, setStreamText] = useState("");
  const [errorState, setErrorState] = useState<null | "timeout" | "offline" | "backend">(null);
  const [retryPayload, setRetryPayload] = useState<RetryPayload | null>(null);

  const [isDesktop, setIsDesktop] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const sendingRef = useRef(false);
  const shouldAutoFollowRef = useRef(true);

  const visibleConversations = useMemo(
    () => conversations.filter(hasConversationContent),
    [conversations],
  );
  const activeConversation = useMemo(
    () => visibleConversations.find((c) => c.id === activeId) ?? null,
    [visibleConversations, activeId],
  );
  const messages = useMemo(() => activeConversation?.messages ?? [], [activeConversation]);
  const busy = stage !== "done";
  const showWelcome = view === "qa" && messages.length === 0 && !busy && !errorState;

  // Dọn các bản ghi rỗng còn sót lại từ phiên bản cũ sau khi localStorage đã hydrate.
  useEffect(() => {
    if (!conversationsHydrated) return;
    setConversations((prev) => {
      const nonEmpty = prev.filter(hasConversationContent);
      return nonEmpty.length === prev.length ? prev : nonEmpty;
    });
  }, [conversationsHydrated, setConversations]);

  const lastResponseMessage = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].response) return messages[i];
    }
    return null;
  }, [messages]);
  const lastResponse: QueryResponse | null = lastResponseMessage?.response ?? null;

  /* --------------------------------- actions -------------------------------- */

  const newConversation = useCallback(() => {
    // Chỉ tạo bản ghi khi người dùng thực sự gửi câu hỏi đầu tiên.
    shouldAutoFollowRef.current = true;
    setActiveId(null);
    setView("qa");
    setEvidenceOpen(false);
    setSelectedSource(null);
    setActiveCitation(null);
    setEvidenceMessageId(null);
    setEvidenceTab("citations");
    setDraft("");
    setErrorState(null);
    setRetryPayload(null);
    setLiveHits([]);
    setStreamText("");
    setTimeout(() => inputRef.current?.focus(), 60);
  }, [setActiveId]);

  const patchConversation = useCallback(
    (id: string, fn: (c: Conversation) => Conversation) => {
      setConversations((prev) => prev.map((c) => (c.id === id ? fn(c) : c)));
    },
    [setConversations],
  );

  const send = useCallback(
    async (
      question: string,
      targetConversationId?: string,
      modeOverride?: QueryMode,
      requestOverride?: QueryRequest,
    ) => {
      const text = question.trim();
      if (!text || busy || sendingRef.current) return;
      sendingRef.current = true;
      shouldAutoFollowRef.current = true;
      const effectiveMode = modeOverride ?? mode;

      const existingId =
        targetConversationId &&
        visibleConversations.some((conversation) => conversation.id === targetConversationId)
          ? targetConversationId
          : activeId && visibleConversations.some((conversation) => conversation.id === activeId)
            ? activeId
            : null;
      const id = existingId ?? uid();
      const createdAt = Date.now();

      setView("qa");
      setSelectedSource(null);
      setActiveCitation(null);
      setEvidenceMessageId(null);
      setEvidenceTab("citations");
      setEvidenceOpen(false);
      setDraft("");
      setErrorState(null);
      setRetryPayload(null);
      setLiveHits([]);
      setStreamText("");

      const userMsg: Message = {
        id: uid(),
        role: "user",
        content: text,
        createdAt,
        mode: effectiveMode,
      };

      setConversations((prev) => {
        const existing = prev.find((conversation) => conversation.id === id);
        if (!existing) {
          return [
            {
              id,
              title: text.slice(0, 48),
              createdAt,
              updatedAt: createdAt,
              pinned: false,
              messages: [userMsg],
            },
            ...prev,
          ];
        }
        return prev.map((conversation) =>
          conversation.id === id
            ? {
                ...conversation,
                title: conversation.messages.length === 0 ? text.slice(0, 48) : conversation.title,
                updatedAt: createdAt,
                messages: [...conversation.messages, userMsg],
              }
            : conversation,
        );
      });
      if (activeId !== id) setActiveId(id);

      const request =
        requestOverride ??
        buildRequest(text, {
          prompt_version:
            effectiveMode === "lookup"
              ? "legal_lookup_v1"
              : effectiveMode === "compare"
                ? "legal_compare_v1"
                : "legal_qa_v1",
          top_k: settings.topK,
          top_n: Math.min(5, settings.topK),
          filters:
            settings.lawName || settings.docId || settings.article
              ? {
                  law_name: settings.lawName || null,
                  doc_id: settings.docId || null,
                  article: settings.article || null,
                }
              : null,
          debug: settings.advanced,
        });

      try {
        const response = await runLegalQuery(request, {
          onStage: setStage,
          onHit: (h) => setLiveHits((prev) => [...prev, h]),
          onToken: setStreamText,
        });

        const assistantMsg: Message = {
          id: uid(),
          role: "assistant",
          content: response.answer,
          createdAt: Date.now(),
          mode: effectiveMode,
          response,
          followUps: suggestFollowUps(text),
        };
        patchConversation(id, (c) => ({
          ...c,
          updatedAt: Date.now(),
          messages: [...c.messages, assistantMsg],
        }));
        setRetryPayload(null);
      } catch (err) {
        const name = (err as Error)?.name;
        setErrorState(
          name === "OfflineError" ? "offline" : name === "AbortError" ? "timeout" : "backend",
        );
        setRetryPayload({
          question: text,
          conversationId: id,
          messageId: userMsg.id,
          mode: effectiveMode,
          request,
        });
      } finally {
        setStage("done");
        setStreamText("");
        setLiveHits([]);
        sendingRef.current = false;
      }
    },
    [
      activeId,
      busy,
      mode,
      patchConversation,
      setActiveId,
      setConversations,
      settings,
      visibleConversations,
    ],
  );

  const savedSourceIds = useMemo(
    () => new Set(savedSources.map((source) => source.id)),
    [savedSources],
  );

  const toggleSavedSource = useCallback(
    (source: SavedLegalSource) => {
      const exists = savedSourceIds.has(source.id);
      setSavedSources((prev) =>
        exists
          ? prev.filter((item) => item.id !== source.id)
          : [{ ...source, savedAt: Date.now() }, ...prev],
      );
      toast.success(exists ? "Đã bỏ lưu nguồn" : "Đã lưu vào Văn bản đã lưu");
    },
    [savedSourceIds, setSavedSources],
  );

  const toggleResponseSources = useCallback(
    (response: QueryResponse) => {
      const sources = response.citations.map(savedSourceFromCitation);
      if (sources.length === 0) return;
      const allSaved = sources.every((source) => savedSourceIds.has(source.id));
      const sourceIds = new Set(sources.map((source) => source.id));
      setSavedSources((prev) => {
        if (allSaved) return prev.filter((source) => !sourceIds.has(source.id));
        const withoutDuplicates = prev.filter((source) => !sourceIds.has(source.id));
        return [
          ...sources.map((source) => ({ ...source, savedAt: Date.now() })),
          ...withoutDuplicates,
        ];
      });
      toast.success(allSaved ? "Đã bỏ lưu các căn cứ" : "Đã lưu các căn cứ của câu trả lời");
    },
    [savedSourceIds, setSavedSources],
  );

  const retryLastQuery = useCallback(() => {
    if (!retryPayload) return;
    setConversations((prev) =>
      prev.map((conversation) =>
        conversation.id === retryPayload.conversationId
          ? {
              ...conversation,
              messages: conversation.messages.filter(
                (message) => message.id !== retryPayload.messageId,
              ),
            }
          : conversation,
      ),
    );
    setActiveId(retryPayload.conversationId);
    setMode(retryPayload.mode);
    setErrorState(null);
    void send(
      retryPayload.question,
      retryPayload.conversationId,
      retryPayload.mode,
      retryPayload.request,
    );
  }, [retryPayload, send, setConversations]);

  const openCitationEvidence = useCallback((messageId: string, cid: string) => {
    setSelectedSource(null);
    setEvidenceMessageId(messageId);
    setEvidenceTab("citations");
    setActiveCitation(cid);
    setEvidenceOpen(true);
    setTimeout(() => {
      document
        .getElementById(`evidence-${cid}`)
        ?.scrollIntoView({ block: "center", behavior: "smooth" });
    }, 120);
  }, []);

  const locateCitationInAnswer = useCallback(
    (cid: string) => {
      if (!evidenceMessageId) return;
      setActiveCitation(cid);
      if (!isDesktop) setEvidenceOpen(false);
      setTimeout(
        () => {
          const inline = document.querySelector<HTMLElement>(
            `[data-citation-message="${evidenceMessageId}"][data-citation-id="${cid}"]`,
          );
          inline?.scrollIntoView({ block: "center", behavior: "smooth" });
          if (inline) {
            inline.classList.add("citation-flash");
            setTimeout(() => inline.classList.remove("citation-flash"), 1300);
          }
        },
        isDesktop ? 80 : 220,
      );
    },
    [evidenceMessageId, isDesktop],
  );

  const renameConversation = useCallback(
    (id: string) => {
      const current = conversations.find((c) => c.id === id);
      const next = window.prompt("Đổi tên cuộc trò chuyện", current?.title ?? "");
      if (next && next.trim()) patchConversation(id, (c) => ({ ...c, title: next.trim() }));
    },
    [conversations, patchConversation],
  );

  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) {
        setActiveId(null);
        setSelectedSource(null);
        setActiveCitation(null);
        setEvidenceMessageId(null);
        setEvidenceTab("citations");
        setEvidenceOpen(false);
      }
      if (activeId === id || retryPayload?.conversationId === id) {
        setErrorState(null);
        setRetryPayload(null);
      }
    },
    [activeId, retryPayload?.conversationId, setActiveId, setConversations],
  );

  const handleViewChange = useCallback((nextView: WorkspaceView) => {
    setView(nextView);
    setSelectedSource(null);
    setActiveCitation(null);
    setEvidenceMessageId(null);
    setEvidenceTab("citations");
    setEvidenceOpen(false);
    setErrorState(null);
    setRetryPayload(null);
    setDrawerOpen(false);
  }, []);

  const handleSelectConversation = useCallback(
    (id: string) => {
      const conversation = visibleConversations.find((item) => item.id === id);
      const recentMode = [...(conversation?.messages ?? [])]
        .reverse()
        .find((message) => message.mode)?.mode;
      shouldAutoFollowRef.current = true;
      setActiveId(id);
      setView("qa");
      if (recentMode) setMode(recentMode);
      setSelectedSource(null);
      setActiveCitation(null);
      setEvidenceMessageId(null);
      setEvidenceTab("citations");
      setEvidenceOpen(false);
      setErrorState(null);
      setRetryPayload(null);
      setDrawerOpen(false);
    },
    [visibleConversations],
  );

  const handleNewConversation = useCallback(() => {
    newConversation();
    setDrawerOpen(false);
  }, [newConversation]);

  const handlePinConversation = useCallback(
    (id: string) =>
      patchConversation(id, (conversation) => ({
        ...conversation,
        pinned: !conversation.pinned,
      })),
    [patchConversation],
  );

  const handleOpenSettings = useCallback(() => setSettingsOpen(true), []);
  const handleCloseDrawer = useCallback(() => setDrawerOpen(false), []);

  /* -------------------------------- shortcuts ------------------------------- */

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1280px)");
    const sync = () => setIsDesktop(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const meta = e.metaKey || e.ctrlKey;
      if (meta && e.shiftKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        newConversation();
      } else if (meta && e.key.toLowerCase() === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      } else if (e.key === "Escape") {
        setEvidenceOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [newConversation]);

  useEffect(() => {
    if (!shouldAutoFollowRef.current) return;
    bottomRef.current?.scrollIntoView({
      behavior: stage === "streaming" ? "auto" : "smooth",
      block: "end",
    });
  }, [activeId, messages.length, stage, streamText]);

  const openDoc = useCallback((doc: LegalDocEntry) => {
    setSelectedSource(savedSourceFromDoc(doc));
    setEvidenceMessageId(null);
    setEvidenceTab("citations");
    setActiveCitation(null);
    setEvidenceOpen(true);
  }, []);

  const selectedEvidence = useMemo(
    () => (selectedSource ? evidenceFromSavedSource(selectedSource) : null),
    [selectedSource],
  );
  const responseForEvidence = useMemo(
    () =>
      evidenceMessageId
        ? (messages.find((message) => message.id === evidenceMessageId)?.response ?? null)
        : lastResponse,
    [evidenceMessageId, lastResponse, messages],
  );
  const evidenceCitations = selectedEvidence
    ? [selectedEvidence.citation]
    : (responseForEvidence?.citations ?? []);
  const evidenceHits = selectedEvidence
    ? [selectedEvidence.hit]
    : (responseForEvidence?.retrieval_hits ?? []);

  const sharedSidebarProps = {
    view,
    onViewChange: handleViewChange,
    conversations: visibleConversations,
    activeId,
    onSelect: handleSelectConversation,
    onNew: handleNewConversation,
    onRename: renameConversation,
    onPin: handlePinConversation,
    onDelete: deleteConversation,
    theme,
    onToggleTheme: toggleTheme,
    onOpenSettings: handleOpenSettings,
  };

  const sidebar = <PersistentSidebar {...sharedSidebarProps} />;
  const mobileSidebar = (
    <Sidebar
      {...sharedSidebarProps}
      collapsed={false}
      inDrawer
      onToggleCollapse={handleCloseDrawer}
    />
  );

  const evidence = (
    <EvidencePanelContent
      citations={evidenceCitations}
      hits={evidenceHits}
      response={selectedSource ? null : responseForEvidence}
      activeCitation={activeCitation}
      onSelectCitation={selectedSource || !evidenceMessageId ? undefined : locateCitationInAnswer}
      savedSourceIds={savedSourceIds}
      onToggleSaved={toggleSavedSource}
      activeTab={evidenceTab}
      onTabChange={setEvidenceTab}
      onClose={() => setEvidenceOpen(false)}
      advanced={settings.advanced}
    />
  );

  return (
    <TooltipProvider delayDuration={250}>
      <div className="flex h-screen h-dvh w-full overflow-hidden bg-background">
        <div className="hidden md:block">{sidebar}</div>

        <Sheet open={drawerOpen} onOpenChange={setDrawerOpen}>
          <SheetContent
            side="left"
            className="w-[272px] max-w-[calc(100vw-24px)] p-0 [&>button]:hidden"
          >
            <SheetTitle className="sr-only">Điều hướng</SheetTitle>
            <div className="h-full">{mobileSidebar}</div>
          </SheetContent>
        </Sheet>

        <div className="flex min-w-0 flex-1 flex-col">
          <TopBar
            title={
              view === "search"
                ? "Tra cứu văn bản"
                : view === "saved"
                  ? "Văn bản đã lưu"
                  : (activeConversation?.title ?? "Cuộc trò chuyện mới")
            }
            evidenceAvailable={Boolean(selectedSource || lastResponse)}
            conversationActionsAvailable={Boolean(activeConversation)}
            onOpenEvidence={() => {
              if (!selectedSource && lastResponseMessage) {
                setEvidenceMessageId(lastResponseMessage.id);
              }
              setEvidenceTab("citations");
              setActiveCitation(null);
              setEvidenceOpen(true);
            }}
            onRename={() => activeId && renameConversation(activeId)}
            onClear={() => activeId && deleteConversation(activeId)}
            onOpenSettings={handleOpenSettings}
            leading={
              <button
                type="button"
                onClick={() => setDrawerOpen(true)}
                aria-label="Mở thanh bên"
                className="inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none md:hidden"
              >
                <Menu className="size-4" aria-hidden />
              </button>
            }
          />

          <div className="flex min-h-0 flex-1">
            <main
              className={`scroll-slim relative flex min-w-0 flex-1 flex-col ${
                showWelcome
                  ? "paper welcome-shell overflow-hidden"
                  : view === "qa"
                    ? "overflow-hidden"
                    : "overflow-y-auto"
              }`}
            >
              {view === "search" && (
                <LegalSearch
                  onOpenEvidence={openDoc}
                  onAsk={(q) => void send(q, undefined, "lookup")}
                />
              )}

              {view === "saved" && (
                <div className="mx-auto w-full max-w-[880px] px-5 py-10">
                  <h2 className="font-serif text-[22px] font-semibold tracking-tight text-foreground">
                    Văn bản đã lưu
                  </h2>
                  <p className="mt-1 text-[13.5px] text-muted-foreground">
                    Các điều khoản bạn đã đánh dấu trong quá trình nghiên cứu.
                  </p>
                  {savedSources.length === 0 ? (
                    <div className="paper mt-6 flex flex-col items-center rounded-xl border border-border px-5 py-14 text-center">
                      <span className="grid size-10 place-items-center rounded-xl bg-surface-2 text-[var(--brand)]">
                        <Bookmark className="size-5" aria-hidden />
                      </span>
                      <p className="mt-3 text-[14px] font-semibold text-foreground">
                        Chưa có văn bản được lưu
                      </p>
                      <p className="mt-1 max-w-[42ch] text-[12.5px] leading-relaxed text-muted-foreground">
                        Mở bảng căn cứ pháp lý và chọn biểu tượng dấu trang để lưu nguồn cần nghiên
                        cứu.
                      </p>
                    </div>
                  ) : (
                    <ul className="mt-6 divide-y divide-border border-t border-b border-border">
                      {savedSources.map((source) => (
                        <li key={source.id} className="group flex items-start gap-3 py-4">
                          <Bookmark
                            className="mt-0.5 size-4 shrink-0 fill-[color-mix(in_oklab,var(--brand)_18%,transparent)] text-[var(--brand)]"
                            aria-hidden
                          />
                          <button
                            type="button"
                            onClick={() => {
                              setSelectedSource(source);
                              setEvidenceMessageId(null);
                              setEvidenceTab("citations");
                              setActiveCitation(null);
                              setEvidenceOpen(true);
                            }}
                            className="min-w-0 flex-1 text-left focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                          >
                            <span className="block font-serif text-[15px] font-semibold text-foreground group-hover:underline">
                              {source.title}
                            </span>
                            <span className="mt-0.5 block text-[12px] text-muted-foreground">
                              {[source.law_name, source.article, source.clause]
                                .filter(Boolean)
                                .join(" · ")}
                            </span>
                            <span className="mt-1.5 line-clamp-2 block text-[13px] leading-relaxed text-muted-foreground">
                              {source.excerpt}
                            </span>
                          </button>
                          <button
                            type="button"
                            onClick={() => toggleSavedSource(source)}
                            aria-label={`Bỏ lưu ${source.title}`}
                            className="inline-flex size-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground opacity-60 transition-[opacity,color,background-color] hover:bg-accent hover:text-[var(--flag)] hover:opacity-100 focus-visible:opacity-100 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                          >
                            <BookmarkX className="size-4" aria-hidden />
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              {view === "qa" && (
                <>
                  {showWelcome ? (
                    <div className="welcome-stage min-h-0 flex-1 overflow-hidden">
                      <WelcomeState onPick={(p) => void send(p)} />
                    </div>
                  ) : (
                    <div
                      className="conversation-scroll scroll-slim min-h-0 flex-1 overflow-x-hidden overflow-y-auto"
                      onScroll={(event) => {
                        const target = event.currentTarget;
                        shouldAutoFollowRef.current =
                          target.scrollHeight - target.scrollTop - target.clientHeight < 120;
                      }}
                    >
                      <div className="mx-auto w-full max-w-[880px] space-y-6 px-5 py-6">
                        {messages.map((m) =>
                          m.role === "user" ? (
                            <UserMessage key={m.id} content={m.content} />
                          ) : (
                            <AssistantAnswer
                              key={m.id}
                              message={m}
                              activeCitation={evidenceMessageId === m.id ? activeCitation : null}
                              onCitation={(citationId) => openCitationEvidence(m.id, citationId)}
                              onOpenEvidence={() => {
                                setSelectedSource(null);
                                setEvidenceMessageId(m.id);
                                setEvidenceTab("citations");
                                setActiveCitation(null);
                                setEvidenceOpen(true);
                              }}
                              onFollowUp={(q) =>
                                void send(q, activeId ?? undefined, m.mode ?? mode)
                              }
                              sourcesSaved={
                                Boolean(m.response?.citations.length) &&
                                (m.response?.citations.every((citation) =>
                                  savedSourceIds.has(savedSourceFromCitation(citation).id),
                                ) ??
                                  false)
                              }
                              onToggleSavedSources={() => {
                                if (m.response) toggleResponseSources(m.response);
                              }}
                            />
                          ),
                        )}

                        {busy && stage !== "streaming" && (
                          <LoadingPipeline stage={stage} hits={liveHits} />
                        )}

                        {stage === "streaming" && streamText && (
                          <AssistantAnswer
                            streaming
                            message={{
                              id: "streaming",
                              role: "assistant",
                              content: streamText,
                              createdAt: Date.now(),
                            }}
                            activeCitation={activeCitation}
                            onCitation={(citationId) =>
                              openCitationEvidence("streaming", citationId)
                            }
                            onOpenEvidence={() => setEvidenceOpen(true)}
                            onFollowUp={() => {}}
                          />
                        )}

                        {errorState === "offline" && (
                          <WarningBanner
                            tone="offline"
                            title="Bạn đang ngoại tuyến"
                            description="Kết nối mạng bị gián đoạn nên không thể truy hồi văn bản pháp luật."
                            onRetry={retryPayload ? retryLastQuery : undefined}
                          />
                        )}
                        {errorState === "timeout" && (
                          <WarningBanner
                            tone="warning"
                            title="Quá thời gian xử lý"
                            description="Hệ thống truy hồi phản hồi chậm hơn dự kiến. Vui lòng thử lại."
                            onRetry={retryPayload ? retryLastQuery : undefined}
                          />
                        )}
                        {errorState === "backend" && (
                          <WarningBanner
                            tone="error"
                            title="Không kết nối được dịch vụ truy hồi"
                            description="Chưa tìm thấy đủ căn cứ trong dữ liệu hiện có để đưa ra kết luận đáng tin cậy."
                            onRetry={retryPayload ? retryLastQuery : undefined}
                          />
                        )}

                        <div ref={bottomRef} />
                      </div>
                    </div>
                  )}
                </>
              )}

              {view === "qa" && (
                <div
                  className={
                    showWelcome
                      ? "welcome-composer z-10 shrink-0 pt-2"
                      : "z-10 mt-auto shrink-0 bg-background/95 pt-3 backdrop-blur"
                  }
                >
                  <div
                    className="mx-auto w-full max-w-[880px] px-5"
                    style={{
                      paddingBottom: showWelcome
                        ? "max(0.75rem, env(safe-area-inset-bottom))"
                        : "max(1rem, env(safe-area-inset-bottom))",
                    }}
                  >
                    <PromptComposer
                      value={draft}
                      onChange={setDraft}
                      onSubmit={() => void send(draft)}
                      mode={mode}
                      onModeChange={setMode}
                      settings={settings}
                      onSettingsChange={setSettings}
                      busy={busy}
                      inputRef={inputRef}
                    />
                    <p className="welcome-legal-note mt-1.5 flex items-center justify-center gap-2 text-center text-[10.5px] text-muted-foreground">
                      <span className="size-1.5 rotate-45 bg-[var(--flag)]" aria-hidden />
                      Kết quả chỉ mang tính tham khảo pháp lý
                    </p>
                  </div>
                </div>
              )}
            </main>

            {/* Desktop evidence panel */}
            <AnimatePresence initial={false}>
              {evidenceOpen && (
                <motion.aside
                  key="evidence"
                  initial={{ width: 0, opacity: 0 }}
                  animate={{ width: 380, opacity: 1 }}
                  exit={{ width: 0, opacity: 0 }}
                  transition={{ duration: 0.2, ease: "easeOut" }}
                  className="hidden shrink-0 overflow-hidden border-l border-border xl:block"
                  aria-label="Căn cứ pháp lý"
                >
                  <div className="h-full w-[380px]">{evidence}</div>
                </motion.aside>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Tablet & mobile evidence panel */}
        <Sheet open={evidenceOpen && !isDesktop} onOpenChange={setEvidenceOpen}>
          <SheetContent
            side="bottom"
            className="h-[82vh] p-0 [&>button]:hidden"
            aria-label="Căn cứ pháp lý"
          >
            <SheetTitle className="sr-only">Căn cứ pháp lý</SheetTitle>
            <div className="h-full">{evidence}</div>
          </SheetContent>
        </Sheet>

        <SettingsDialog
          open={settingsOpen}
          onOpenChange={setSettingsOpen}
          settings={settings}
          onSettingsChange={setSettings}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
        <Toaster position="bottom-center" />
      </div>
    </TooltipProvider>
  );
}
