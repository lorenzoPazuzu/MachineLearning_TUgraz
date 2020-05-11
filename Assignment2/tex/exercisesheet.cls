\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{exercisesheet}[2020/03/20 Template for solving exercise sheets for university]


\def\immatriculationnumber#1{\def\@immatriculationnumber{#1}}
\newcommand{\printimmatriculationnumber}{\@immatriculationnumber}

\def\subject#1{\def\@subject{#1}}
\newcommand{\printsubject}{\@subject}

\def\semester#1{\def\@semester{#1}}
\newcommand{\printsemester}{\@semester}

\def\sheetnumber#1{\def\@sheetnumber{#1}}
\newcommand{\printsheetnumber}{\@sheetnumber}

\newcommand{\exstring}{Aufgabe}
\newcommand{\sheetstring}{Exercise sheet}
\newcommand{\solstring}{Solution}
\newcommand{\lang}{english}
\newcommand{\bystring}{Solution by}
\LoadClass{article}




\DeclareOption{english}{
\renewcommand{\exstring}{Exercise}
\renewcommand{\sheetstring}{Exercise sheet}
\renewcommand{\solstring}{Solution}
\renewcommand{\lang}{english}
\renewcommand{\bystring}{Solution by}
}

\DeclareOption{german}{
\renewcommand{\exstring}{Aufgabe}
\renewcommand{\sheetstring}{Aufgabenblatt}
\renewcommand{\solstring}{Lösung}
\renewcommand{\lang}{german}
\renewcommand{\bystring}{Lösung von}
}
\ProcessOptions\relax

\RequirePackage[\lang]{babel}

\RequirePackage{geometry}
\geometry{
  left=2.5cm,
  right=1.5cm,
  top=2cm,
  bottom=3cm,
  bindingoffset=5mm
}

%exercise with an additional name
\newcounter{exercise}
\newenvironment{nexercise}[1]{
\refstepcounter{exercise}\par\textbf{\exstring~\theexercise.}  \textbf{(#1)}}
{}

\newenvironment{exercise}{
\refstepcounter{exercise}\par\textbf{\exstring~\theexercise.}}
{}

%execise with custom numbering
\newenvironment{nuexercise}[1]{
\par\textbf{\exstring~#1.}}
{}

\newenvironment{nnexercise}[2]{
\par\textbf{\exstring~#1.} \textbf{(#2)}}
{}

\newenvironment{solution}{\par\textit{\solstring.} }{}

\newcommand{\makedocumentheader}{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.8em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@makedocumentheader
      \else
        \twocolumn[\@makedocumentheader]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@makedocumentheader
      \noindent\rule{\textwidth}{.5pt}
      \vskip 1.5em
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}
\def\@makedocumentheader{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
  \let \footnote \thanks
    {\LARGE \@subject\\\@semester \par}%
    {\large \textbf{\sheetstring~\@sheetnumber} \par}
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \bystring\\\@author\\\@immatriculationnumber
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par}
